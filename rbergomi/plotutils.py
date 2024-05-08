"""
Utilities for plotting
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_fit(model, parCalib, k_obs, T_obs, iv_obs):
    # Get the calibrated implied volatilities
    iv_fit = model.Eval(parCalib, k_obs, T_obs)
    uniqT = np.unique(T_obs)
    plt.figure(1, figsize=(12, 12))
    j = 1

    # Plot the ATM Vols first
    plt.subplot(4, 3, j)
    idx_atm = np.abs(k_obs) < 0.01  # ATM contracts
    plt.plot(T_obs[idx_atm], iv_obs[idx_atm], "b", label="Observed")
    plt.plot(T_obs[idx_atm], iv_fit[idx_atm], "--r", label="Fit")
    plt.title("ATM Term Structure")
    plt.xlabel("Maturity")
    plt.ylabel("Implied volatility")
    plt.legend()
    j = j + 1

    # Plot the Slices one by one.
    iList = np.arange(0, len(uniqT))
    for i in iList:
        plt.subplot(4, 3, j)
        idxT = T_obs == uniqT[i]
        plt.plot(k_obs[idxT], iv_obs[idxT], "b", label="Observed")
        plt.plot(k_obs[idxT], iv_fit[idxT], "--r", label="Fit")
        plt.title("Maturity=%1.3f " % uniqT[i])
        plt.xlabel("Log-moneyness")
        plt.ylabel("Implied volatility")
        plt.legend()
        j = j + 1

    plt.tight_layout()
    plt.show()
