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
