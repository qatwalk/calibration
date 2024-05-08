"""Example of calibrating SVI model to market data."""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from NeuralNetworkPricing import NeuralNetworkPricer

DIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.path.dirname(DIR)
sys.path.append(ROOTDIR)
from utils.svi import svi  # noqa: E402


def fit_bergomi():
    weights_folder = DIR + "\\nn_weights"
    contracts_folder = DIR
    model = NeuralNetworkPricer(contracts_folder, weights_folder, "rbergomi")

    svi_file = ROOTDIR + "\\data\\spx_svi_2005_09_15.csv"
    svi_df = pd.read_csv(svi_file, parse_dates=["date"])
    expirations = list(svi_df["date"][1:-2])  # Skip the last two expirations

    # candidate log strikes for any expiration before filtering
    k_candidates = np.arange(-0.5, 0.5, 0.02).reshape(-1, 1)

    k_orig = []
    T_orig = []
    k_obs = []
    T_obs = []
    iv_obs = []

    for i, exp in enumerate(expirations):
        params = svi_df[svi_df["date"] == exp].iloc[0]

        t_candidates = k_candidates * 0 + params["texp"]
        idxKeep = model.AreContractsInDomain(k_candidates, t_candidates)
        k_obs_sub = k_candidates[idxKeep, :]
        T_obs_sub = t_candidates[idxKeep, :]

        w = svi(params, k_obs_sub)
        svi_iv_sub = np.sqrt(w / params["texp"])
        k_orig.extend(k_candidates)
        T_orig.extend(t_candidates)
        k_obs.extend(k_obs_sub)
        T_obs.extend(T_obs_sub)
        iv_obs.extend(svi_iv_sub)

    # Turn into column vectors for the model
    iv_obs = np.array(iv_obs).reshape(-1, 1)
    k_obs = np.array(k_obs).reshape(-1, 1)
    T_obs = np.array(T_obs).reshape(-1, 1)

    # Set parameter bounds:
    # Remark: Optimizer occassionally goes beyond the specified bounds. Thus we make the bounds slightly more narrow.
    modelbounds = []
    eps = pow(10, -6)
    for i in range(0, len(model.lb)):
        modelbounds.append([model.lb[i] + eps, model.ub[i] - eps])

    # Choose a parameteric shape for the forward variance curve with two paramters:
    txi = model.Txi

    def param_full(x):
        xi = x[4] + np.exp(-x[5] * txi) * (x[3] - x[4])
        return np.concatenate((x[:3], xi))

    bounds = modelbounds[0:6]
    bounds[5] = [0, 10]  # Bounds for the decay parameter
    par0 = np.array([0.2, 1.6, -0.7, 0.02, 0.05, 5])

    # Define the error function:
    def err_fun(parEval):
        return np.sum(
            pow(
                iv_obs
                - model.Eval(param_full(parEval).reshape(-1, 1), k_obs, T_obs),
                2,
            )
        )

    # Optimize:
    res = minimize(err_fun, par0, method="L-BFGS-B", bounds=bounds)
    print("Optimisation message: ", res.message)
    print(res.x)
    parCalib = param_full(res.x).reshape(-1, 1)

    print(parCalib.reshape(-1))


if __name__ == "__main__":
    # Plot the fit:
    fit_bergomi()
