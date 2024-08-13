import sys
import os
from base_fns import get_local_folder

DIR = get_local_folder()
ROOTDIR = os.path.dirname(DIR)
sys.path.append(ROOTDIR)

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from functools import partial

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
    
    # DataFrames outside the loop
    abs_strikes_df = pd.DataFrame(abs_strikes, index=svi_df.index, columns=log_strikes)
    implied_vols_df = pd.DataFrame(implied_vols, index=svi_df.index, columns=abs_strikes_df.iloc[0])
    
    return implied_vols_df, abs_strikes_df

# Data
svi_file = ROOTDIR + "\\data\\spx_svi_2005_09_15.csv"
svi_df = pd.read_csv(svi_file, parse_dates=["date"])

svi_df.set_index("date", inplace=True)
log_strikes = np.linspace(-0.5, 0.5, 100)  # Define log strikes
forward_price = 1200  # Assume a constant forward price
risk_free_rate = 0.01  # Example risk-free rate

# implied volatilities and absolute strikes
implied_vols_df, abs_strikes_df = calculate_implied_vols(svi_df, log_strikes, forward_price)

# BLACK-SCHOLES MODEL
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

# Heston model characteristic function
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

# integrals for pricing using the Fourier transform
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

# main function to calculate the option price
def price_vanilla_call(
    K,  # strike
    T,  # option maturity in years
    S0, # current spot price
    r,  # risk-free rate,0.01
    v0, # Intial Variance, 0.0174
    mu, # Drift rate , 0.01
    kappa, # Speed of Mean reversion ,1.35253
    theta, # Long term Mean Variance,0.0354
    sigma, # Volatility
    rho # Correlation ,-0.7165
):
    mu = r  # Drift rate, often the risk-free rate
    cf_reduced = partial(cf_heston,t=T,v0=v0,mu=mu,theta=theta,sigma=sigma,kappa=kappa,rho=rho,)
    k = np.log(K / S0)  # log strike
    price = S0 * np.exp((mu - r) * T) * Q1(k, cf_reduced) - K * np.exp(-r * T) * Q2(k, cf_reduced)
    return price
''' 
            "INITIAL_VAR": 0.0174,
            "LONG_VAR": 0.0354,
            "VOL_OF_VAR": 0.3877,
            "MEANREV": 1.3253,
            "CORRELATION": -0.7165,
'''

# Heston option prices
option_prices_heston = []
for date in svi_df.index:
    prices = []
    for strike in abs_strikes_df.loc[date]:
        price = price_vanilla_call(strike, svi_df.loc[date, 'texp'], forward_price, risk_free_rate, 0.0174, 0.01, 1.3253, 0.0354, svi_df.loc[date, 'sig'], -0.7165)
        prices.append(price)
    option_prices_heston.append(prices)

option_prices_heston_df = pd.DataFrame(option_prices_heston, index=svi_df.index, columns=abs_strikes_df.columns)

# results for comparison
comparison_df = pd.DataFrame(index=svi_df.index, columns=['Strike', 'BS_Price', 'Heston_Price'])

# comparison DataFrame with prices for the first absolute strike
comparison_df['Strike'] = abs_strikes_df.iloc[:, 0]  # strike
comparison_df['BS_Price'] = call_prices_df.iloc[:, 0]  #  BS prices
comparison_df['Heston_Price'] = option_prices_heston_df.iloc[:, 0]  #  Heston prices

# Output
print(comparison_df)
