from math import log, sqrt, exp, erf
from datetime import datetime, timezone
from mpl_toolkits.mplot3d import Axes3D
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

def get_r(maturity = "DGS1", lookback_days = 14):
    end = dt.datetime.today()
    start = end - dt.timedelta(days = lookback_days)
    df = pdr.DataReader(maturity, "fred", start, end)
    return float(df.dropna().iloc[-1, 0]) / 100.0

def filter_invalid(df, r = 0.0):
    S = df["S"].to_numpy()
    K = df["strike"].to_numpy()
    T = df["T"].to_numpy()
    q = df["q"].to_numpy()
    typ = df["type"].to_numpy()
    mid = df["mid"].to_numpy()
    time_ok = T >= 1.0 / 365.0
    spot_ok = S > 0.0
    price_ok = mid > 0.0
    moneyness_ok = np.abs(np.log(K / S)) <= 3.0
    lower = np.empty_like(mid, dtype=float)
    upper = np.empty_like(mid, dtype=float)
    for i in range(len(df)):
        l, u = noarb_bounds(S[i], K[i], r, q[i], T[i], typ[i])
        lower[i] = l
        upper[i] = u
    bounds_ok = (mid >= lower) & (mid <= upper)
    mask = time_ok & spot_ok & price_ok & moneyness_ok & bounds_ok
    return df.loc[mask].reset_index(drop = True)

def fetch_chain(tickers, r = 0.0):
    if isinstance(tickers, str):
        tickers = [tickers]
    snapshot = datetime.now(timezone.utc)
    print(f"Snapshot (UTC): {snapshot.isoformat(timespec = 'seconds')}")
    rows = []
    for tk in tickers:
        Tkr = yf.Ticker(tk)
        spot_hist = Tkr.history(period = "1d")
        spot = float(spot_hist.iloc[-1]["Close"])
        if "dividendYield" in Tkr.info and Tkr.info["dividendYield"] is not None:
            q = float(Tkr.info["dividendYield"])
        elif hasattr(Tkr, "fast_info") and "dividendYield" in Tkr.fast_info:
            q = float(Tkr.fast_info["dividendYield"])
        else:
            q = 0.0
        for expiry in Tkr.options:
            chain = Tkr.option_chain(expiry)
            expiry_dt = pd.to_datetime(expiry, utc = True)
            T = (expiry_dt - pd.Timestamp(snapshot)).total_seconds() / (365.25 * 24 * 3600)
            F = spot * np.exp((r - q) * T)
            for typ, df in (("call", chain.calls), ("put", chain.puts)):
                strike = df["strike"].astype(float)
                bid = pd.to_numeric(df["bid"])
                ask = pd.to_numeric(df["ask"])
                last = pd.to_numeric(df["lastPrice"])
                mid = np.where(np.isfinite(bid) & np.isfinite(ask), 0.5 * (bid + ask), last)
                base = pd.DataFrame({
                    "ticker": tk,
                    "type": typ,
                    "strike": strike,
                    "mid": mid.astype(float),
                    "S": float(spot),
                    "T": T,
                    "q": q,
                    "F": F
                })
                rows.append(base)
    df = pd.concat(rows, ignore_index = True)
    df_filtered = filter_invalid(df, r = r)
    print(f"Total: {len(df)}, Valid: {len(df_filtered)}")
    return df_filtered

def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def d1(S, K, r, q, sigma, T):
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

def d2(S, K, r, q, sigma, T):
    return d1(S, K, r, q, sigma, T) - sigma * sqrt(T)

def black_scholes(S, K, r, q, sigma, T, type):
    if type == "call":
        return S * exp(-q * T) * phi(d1(S, K, r, q, sigma, T)) - K * exp(-r * T) * phi(d2(S, K, r, q, sigma, T))
    else:
        return K * exp(-r * T) * phi(-d2(S, K, r, q, sigma, T)) - S * exp(-q * T) * phi(-d1(S, K, r, q, sigma, T))

def vega(S, K, r, q, sigma, T):
    d1v = d1(S, K, r, q, sigma, T)
    pdf = np.exp(-0.5 * d1v * d1v) / np.sqrt(2.0 * np.pi)
    return S * np.exp(-q * T) * pdf * sqrt(T)

def brenner_subrahmanyam(S, T, price):
    if T <= 0 or S <= 0:
        return 0.2
    return np.clip(np.sqrt(2.0 * np.pi / T) * (price / S), 1e-4, 3.0)

def noarb_bounds(S, K, r, q, T, type):
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if type == "call":
        lower = max(0.0, S * disc_q - K * disc_r); upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q); upper = K * disc_r
    return lower, upper

def iv_Newton(S, K, r, q, T, price, type, sigma0 = None, tol = 1e-8, max_iter = 20):
    lower, upper = noarb_bounds(S, K, r, q, T, type)
    if (price - lower) <= 1e-10 or (upper - price) <= 1e-10:
        return 1e-6
    sigma = sigma0 if sigma0 is not None else brenner_subrahmanyam(S, T, price)
    sigma = float(np.clip(sigma, 1e-4, 3.0))
    for _ in range(max_iter):
        model = black_scholes(S, K, r, q, sigma, T, type)
        diff = model - price
        if abs(diff) < tol:
            return float(sigma)
        v = vega(S, K, r, q, sigma, T)
        step = float(np.clip(diff / v, -0.5, 0.5))
        sigma = float(np.clip(sigma - step, 1e-6, 5.0))
    return np.nan

def iv_bisect(S, K, r, q, T, price, type, lo = 1e-6, hi = 5.0, tol = 1e-8, max_iter = 100):
    f_lo = black_scholes(S, K, r, q, lo, T, type) - price
    f_hi = black_scholes(S, K, r, q, hi, T, type) - price
    tries = 0
    while f_lo * f_hi > 0 and hi < 10.0 and tries < 5:
        hi *= 1.5
        f_hi = black_scholes(S, K, r, q, hi, T, type) - price
        tries += 1
    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = black_scholes(S, K, r, q, m, T, type) - price
        if abs(fm) < tol or (b - a) < 1e-8:
            return float(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))

def compute_iv(df, r = 0.0):
    out = df.copy().sort_values(["ticker", "T", "strike"]).reset_index(drop = True)
    ivs = []
    last_ticker = None
    last_T = None
    last_sigma = None
    for _, row in out.iterrows():
        if row["ticker"] != last_ticker or row["T"] != last_T:
            last_sigma = None
            last_ticker = row["ticker"]
            last_T = row["T"]
        S, K, q, T, P, typ = row["S"], row["strike"], row["q"], row["T"], row["mid"], row["type"]
        lower, upper = noarb_bounds(S, K, r, q, T, typ)
        rel = (P - lower) / max(upper - lower, 1e-12)
        use_newton = (0.02 < rel < 0.98) and (T > 5/365) and (abs(np.log(K / S)) < 1.5)

        if use_newton:
            iv = iv_Newton(S, K, r, q, T, P, typ, sigma0 = last_sigma)
            if not np.isfinite(iv):
                iv = iv_bisect(S, K, r, q, T, P, typ)
        else:
            iv = iv_bisect(S, K, r, q, T, P, typ)
        ivs.append(iv)
        if np.isfinite(iv):
            last_sigma = iv
    out["iv"] = np.array(ivs, dtype = float)
    return out

def plot_iv_smile(iv_df, target = 30, tickers = None):
    if isinstance(tickers, str):
        tickers = [tickers]
    idx = (iv_df["T"] - target / 365.0).abs().idxmin()
    expiry_T = iv_df.loc[idx, "T"]
    subset = iv_df[iv_df["T"] == expiry_T]
    if tickers is not None:
        subset = subset[subset["ticker"].isin(tickers)]
    fig, ax = plt.subplots(figsize = (10, 7))
    for tk in subset["ticker"].unique():
        for opt_type in ["call", "put"]:
            tmp = subset[(subset["ticker"] == tk) & (subset["type"] == opt_type)]
            if not tmp.empty:
                ax.plot(tmp["strike"], tmp["iv"], marker = "o", linestyle = "", label = f"{tk} {opt_type}s")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Implied Volatility Smile, T ≈ {expiry_T*365:.0f} days")
    ax.legend()
    fig.tight_layout()

def plot_iv_surface(iv_df, tickers = None):
    if isinstance(tickers, str):
        tickers = [tickers]
    if tickers is not None:
        iv_df = iv_df[iv_df["ticker"].isin(tickers)]
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection = "3d")
    for tk in iv_df["ticker"].unique():
        subset = iv_df[iv_df["ticker"] == tk]
        mask_call = subset["type"] == "call"
        mask_put  = subset["type"] == "put"
        ax.scatter(subset["T"][mask_call], (subset["strike"] / subset["F"])[mask_call], subset["iv"][mask_call], s = 8, label = f"{tk} calls")
        ax.scatter(subset["T"][mask_put], (subset["strike"] / subset["F"])[mask_put], subset["iv"][mask_put], s = 8, label = f"{tk} puts")
    ax.set_xlabel("T (years)")
    ax.set_ylabel("K / F")
    ax.set_title("Implied Volatility Surface")
    ax.legend()
    ax.invert_yaxis()

def atm_iv(iv_df, method = "nearest", inplace = True):
    if inplace:
        out = iv_df
    else:
        out = iv_df.copy()
    out["K_atm"] = np.nan
    out["iv_atm"] = np.nan
    for (tk, opt_type, T), g in out.groupby(["ticker", "type", "T"]):
        if g.empty:
            continue
        F_T = float(g["F"].iloc[0])
        if method == "nearest" or len(g) < 2:
            idx = (g["strike"] - F_T).abs().idxmin()
            K_atm = float(out.loc[idx, "strike"])
            iv_atm = float(out.loc[idx, "iv"])
        else:
            x = (g["strike"] / F_T).to_numpy()
            y = g["iv"].to_numpy()
            order = np.argsort(x)
            x, y = x[order], y[order]
            if np.isfinite(x).sum() >= 2:
                iv_atm = float(np.interp(1.0, x, y))
                K_atm = float(F_T)
            else:
                idx = (g["strike"] - F_T).abs().idxmin()
                K_atm = float(out.loc[idx, "strike"])
                iv_atm = float(out.loc[idx, "iv"])
        out.loc[g.index, "K_atm"] = K_atm
        out.loc[g.index, "iv_atm"] = iv_atm
    if not inplace:
        return out
    
def plot_term_structure(iv_df, tickers = None):
    if "iv_atm" not in iv_df.columns:
        raise ValueError("iv_atm column is missing; run atm_iv(...) first.")
    if isinstance(tickers, str):
        tickers = [tickers]
    df = iv_df if tickers is None else iv_df[iv_df["ticker"].isin(tickers)]
    fig, ax = plt.subplots(figsize = (8, 5))
    for tk in df["ticker"].unique():
        for opt_type in ["call", "put"]:
            g = df[(df["ticker"] == tk) & (df["type"] == opt_type)]
            if g.empty:
                continue
            pts = g[["T", "iv_atm"]].drop_duplicates().sort_values("T")
            ax.plot(pts["T"], pts["iv_atm"], marker = "o", linestyle = "-",
                    label = f"{tk} {opt_type}")
    ax.set_xlabel("T (years)")
    ax.set_ylabel("ATM IV")
    ax.set_title("ATM IV Term Structure")
    ax.legend()
    fig.tight_layout()
