"""Pricing functions used by the Heston thesis workflow."""

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq
from scipy.stats import norm

from config import (
    DIVIDEND_YIELD,
    MATURITIES,
    RISK_FREE_RATE,
    SPOT_PRICE,
    STRIKES,
    VALUATION_DATE,
)


def heston_price(S, K, T, r, kappa, theta, sigma, rho, v0):
    """Price a European call option under the Heston model."""
    today = ql.Date(VALUATION_DATE.day, VALUATION_DATE.month, VALUATION_DATE.year)
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(round(T * 365), ql.Days)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, DIVIDEND_YIELD, ql.Actual365Fixed())
    )

    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(engine)
    return option.NPV()


def test_heston_pricer():
    """Run the pricing examples and Black-Scholes verification from the notebook."""
    price = heston_price(
        S=SPOT_PRICE,
        K=SPOT_PRICE,
        T=1.0,
        r=RISK_FREE_RATE,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
    )
    print(f"Heston call price: {price:.4f}")
    print(
        heston_price(
            S=SPOT_PRICE,
            K=SPOT_PRICE,
            T=5,
            r=RISK_FREE_RATE,
            kappa=2.0,
            theta=0.04,
            sigma=0.5,
            rho=-0.7,
            v0=0.04,
        )
    )

    S, K, T, r, vol = 100, 100, 1.0, 0.03, 0.20
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    heston_test = heston_price(
        S=100,
        K=100,
        T=1.0,
        r=0.03,
        kappa=2.0,
        theta=0.04,
        sigma=0.001,
        rho=0.0,
        v0=0.04,
    )
    print(f"Black-Scholes price: {bs_price:.4f}")
    print(f"Heston price:        {heston_test:.4f}")
    print(f"Difference:          {abs(bs_price - heston_test):.6f}")


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price given volatility sigma."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    """Find Black-Scholes implied volatility for a given option price."""
    try:
        iv = brentq(
            lambda sigma: bs_call_price(S, K, T, r, sigma) - price,
            0.01,
            5.0,
        )
        return iv
    except:
        return np.nan


def surface_for(kappa, theta, sigma, rho, v0):
    """Calculate a complete implied-volatility surface for Heston parameters."""
    today = ql.Date(VALUATION_DATE.day, VALUATION_DATE.month, VALUATION_DATE.year)
    ql.Settings.instance().evaluationDate = today
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(SPOT_PRICE))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, RISK_FREE_RATE, ql.Actual365Fixed())
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, DIVIDEND_YIELD, ql.Actual365Fixed())
    )
    option_type = ql.Option.Call

    options_list = []
    # Keep this order aligned with the feature rows saved by data_generation.py.
    for T in MATURITIES:
        for K in STRIKES:
            maturity_date = today + ql.Period(round(T * 365), ql.Days)
            payoff = ql.PlainVanillaPayoff(option_type, K)
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            options_list.append((option, K, T))

    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    results = []
    for option, K, T in options_list:
        option.setPricingEngine(engine)
        price = option.NPV()
        iv = implied_vol(price, SPOT_PRICE, K, T, RISK_FREE_RATE)
        results.append(iv)
    return results