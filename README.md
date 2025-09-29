# Implied Volatility Surface

Implied volatility (IV) is the volatility level that, when plugged into the Black–Scholes formula, reproduces the observed option prices for an asset. IV is valuable because it reflects the market’s forward-looking expectations of uncertainty, rather than just historical price variation. In this project, daily option chain data is collected from Yahoo Finance, cleaned with no-arbitrage checks, and used to extract implied volatilities by numerically inverting the Black–Scholes model with Newton–Raphson and bisection methods. Comparing IV across strikes and maturities produces volatility “smiles” and “skews” for fixed expiries, and stacking these together yields a full 3D volatility surface.

The project is organized into two main directories. The `/notebooks` folder contains the `implied_volatility_surface.ipynb` notebook. The `/src` folder holds the source code file `utils.py` with functions for data fetching, filtering, and IV computation.

---

## What It Does

- Fetches daily stock option chain data from Yahoo Finance for a set of tickers.
- Filters out incomplete quotes and entries failing no-arbitrage bounds.
- Computes implied volatilities using Newton–Raphson and bisection methods.
- Plots IV smiles for fixed maturities.
- Builds 3D implied volatility surfaces over strikes and maturities.
- Tracks ATM IV term structures to study volatility expectations.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/your-username/implied-volatility-surface>
   cd implied-volatility-surface

---

## Contents of the Notebook

Notebook `/notebooks/implied_volatility_surface.ipynb` is divided into two parts. The first section constructs and visualizes implied volatility smiles and surfaces across strikes and maturities for a selection of major tech stocks. The second section computes ATM IVs for these stocks, enabling us to track their term structures across maturities and explore how they can inform volatility forecasts.

---

## Contents of the `/src` folder

The **`utils.py`** file provides functions for fetching option chain data, filtering invalid quotes, computing implied volatilities, and visualizing results.

- **`get_r(maturity = "DGS1", lookback_days = 14)`**:  
  &nbsp;&nbsp;&nbsp;Fetches the most recent U.S. Treasury yield from FRED for a given maturity and returns it as the risk-free rate. By default, **maturity = "DGS1"** (1-year yield) serves as a simple all-purpose choice for Black–Scholes inversion, while **lookback_days** (default 14) averages over recent days to reduce noise.

- **`fetch_chain(tickers, r = 0.0)`**:  
  &nbsp;&nbsp;&nbsp;Downloads option chains for one or more tickers from Yahoo Finance, extracts mid prices, computes forward prices $F = S\ e^{(r - q)T}$ and filters the data. Returns the result as a pandas DataFrame. The **tickers** argument can be a single ticker or a list of tickers.

- **`filter_invalid(df, r = 0.0)`**:  
  &nbsp;&nbsp;&nbsp;Removes option quotes that fail basic validity checks (time-to-maturity, price positivity, moneyness bounds, and no-arbitrage limits). Applied automatically inside the `fetch_chain` function.

- **`black_scholes(...)`**, **`phi(...)`**, **`d1(...)`**, **`d2(...)`**, **`vega(...)`**:  
  &nbsp;&nbsp;&nbsp;Implement the Black–Scholes pricing formula and supporting components.

- **`brenner_subrahmanyam(S, T, V)`**:  
  &nbsp;&nbsp;&nbsp;Provides an initial guess for implied volatility using the Brenner–Subrahmanyam approximation
  
  $$
  \sigma \approx \sqrt{\tfrac{2\pi}{T}} \cdot \frac{\text{V}}{S}
  $$

- **`noarb_bounds(S, K, r, q, T, type)`**:  
  &nbsp;&nbsp;&nbsp;Computes theoretical lower and upper bounds for call and put prices to enforce no-arbitrage.

- **`iv_Newton(...)`**:  
  &nbsp;&nbsp;&nbsp;Computes implied volatility using the Newton–Raphson method with vega-based updates.

- **`iv_bisect(...)`**:  
  &nbsp;&nbsp;&nbsp;Computes implied volatility using the bisection method, robust when Newton–Raphson fails.

- **`compute_iv(df, r = 0.0)`**:  
  &nbsp;&nbsp;&nbsp;Applies root-finding methods to compute implied volatilities for an entire option chain DataFrame.

- **`plot_iv_smile(iv_df, target = 30, tickers = None)`**:  
  &nbsp;&nbsp;&nbsp;Plots the implied volatility smile for a maturity closest to the given target (in days), separated by ticker and option type.

- **`plot_iv_surface(iv_df, tickers = None)`**:  
  &nbsp;&nbsp;&nbsp;Creates a 3D scatter plot of implied volatility as a function of maturity and strike/forward ratio.

- **`atm_iv(iv_df, method = "nearest", inplace = True)`**:  
  &nbsp;&nbsp;&nbsp;Computes at-the-money implied volatility for each ticker, maturity, and option type, and stores results in new columns.

- **`plot_term_structure(iv_df, tickers = None)`**:  
  &nbsp;&nbsp;&nbsp;Plots the term structure of ATM IV across maturities for selected tickers and option types.

