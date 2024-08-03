import sys
import os
from base_fns import get_local_folder

DIR = get_local_folder()
ROOTDIR = os.path.dirname(DIR)
sys.path.append(ROOTDIR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

'''--------------------------------------------------------------------------------------------------'''

# SVI Function
def svi(params, k):
    """Get total variance from SVI params and log strikes."""
    k_m = k - params["m"]
    discr = np.sqrt(k_m**2 + params["sig"] ** 2)
    w = params["a"] + params["b"] * (params["rho"] * k_m + discr)
    return w


# IMPLIED VOLATILITY
def calculate_implied_vols(svi_df, log_strikes, forward_price):
    implied_vols = []
    abs_strikes = []

    for _, row in svi_df.iterrows():
        params = row.to_dict()
        total_variance = svi(params, log_strikes)
        implied_vol = np.sqrt(total_variance / params["texp"])
        abs_strike = forward_price * np.exp(log_strikes)
        
        implied_vols.append(implied_vol)
        abs_strikes.append(abs_strike)
    
    # Create DataFrames outside the loop
    abs_strikes_df = pd.DataFrame(abs_strikes, index=svi_df.index, columns=log_strikes)
    implied_vols_df = pd.DataFrame(implied_vols, index=svi_df.index, columns=abs_strikes_df.iloc[0])
    
    return implied_vols_df, abs_strikes_df

'''-----------------------------------------------------------------------------------------------------------------'''
# Data
svi_file = ROOTDIR + "\\data\\spx_svi_2005_09_15.csv"
svi_df = pd.read_csv(svi_file, parse_dates=["date"])
expirations = list(svi_df["date"][1:-2])

svi_df.set_index("date", inplace=True)
log_strikes = np.linspace(-0.5, 0.5, 100)  # Define log strikes
forward_price = 1200  # Assume a constant forward price
risk_free_rate = 0.01  # Example risk-free rate

# Calculate implied volatilities and absolute strikes
implied_vols_df, abs_strikes_df = calculate_implied_vols(svi_df, log_strikes, forward_price)


# BLACKSCHOLES MODEL
def black_scholes_price(forward, strike, time_to_exp, vol, risk_free_rate):
    """Calculate Black-Scholes call option price."""
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * time_to_exp) / (vol * np.sqrt(time_to_exp))
    d2 = d1 - vol * np.sqrt(time_to_exp)
    call_price = np.exp(-risk_free_rate * time_to_exp) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return call_price

# Black-Scholes call prices
call_prices = []
for date in implied_vols_df.index:
    prices = []
    for strike, vol in zip(abs_strikes_df.loc[date], implied_vols_df.loc[date]):
        price = black_scholes_price(forward_price, strike, svi_df.loc[date, "texp"], vol, risk_free_rate)
        prices.append(price)
    call_prices.append(prices)

call_prices_df = pd.DataFrame(call_prices, index=svi_df.index, columns=abs_strikes_df.columns)



'''-----------------------------------------------------------------------------------------------------------'''
# HESTON MODEL PRICE_ CHARACTERSTIC_FUNCTION
def cf_heston(u, t, v0, mu, kappa, theta, sigma, rho):
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta)
        / (sigma**2)
        * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2)
        * (xi - d)
        * (1 - np.exp(-d * t))
        / (1 - g2 * np.exp(-d * t))
    )
    return cf

def Q1(k, cf, right_lim=1000):
    def integrand(u):
        return np.real(
            (np.exp(-u * k * 1j) / (u * 1j))
            * cf(u - 1j)
            / cf(-1.0000000000001j)
        )
    return 1 / 2 + 1 / np.pi * quad(integrand, 0, right_lim, limit=2000)[0]

def Q2(k, cf, right_lim=1000):
    def integrand(u):
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))
    return 1 / 2 + 1 / np.pi * quad(integrand, 0, right_lim, limit=2000)[0]

def price_vanilla_call(
    K,  # strike
    T,  # option maturity in years
    S0, # current spot price
    r,  # risk-free rate
    v0, mu, kappa, theta, sigma, rho
):
    """the price of a Vanilla European Option using Heston Model characteristic function."""
    
    mu = r  # Assuming drift equals risk-free rate if no dividend yield

    cf_reduced = partial(
        cf_heston,
        t=T,
        v0=v0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        kappa=kappa,
        rho=rho,
    )
    k = np.log(K / S0)  # log strike
    price = S0 * np.exp((mu - r) * T) * Q1(
        k, cf_reduced
    ) - K * np.exp(-r * T) * Q2(k, cf_reduced)
    return price

# Example usage
K =         # Strike price
T =           # Time to maturity
S0 =        # Current spot price
r = 0.01       # Risk-free interest rate
v0 =       # Initial variance
mu = 0.01      # Drift term = often set equal to risk-free rate
kappa =    # Rate of mean reversion
theta =   # Long-term variance
sigma =    # Volatility of variance
rho =     # Correlation 

call_price = price_vanilla_call(K, T, S0, r, v0, mu, kappa, theta, sigma, rho)




