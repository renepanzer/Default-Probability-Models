#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default probability models for a single distressed firm.
Usage:
    python scripts/run_analysis.py                 # licensed panels in data/raw
    python scripts/run_analysis.py --sample        # synthetic panels in data/sample
"""

import argparse
import sys
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrisk.merton import MertonModel
from creditrisk.barrier import BlackCoxBarrier
from creditrisk.jump_diffusion import JumpDiffusion
from creditrisk.altman import Altman


#====================================================================================
# File locations. Relative to the repository root.

raw_paths = {
    "prices": "data/raw/TUP_prices.csv",
    "debt": "data/raw/TUP_debt.csv",
    "fundamentals": "data/raw/TUP_Fundamentals.csv"}

sample_paths = {
    "prices": "data/sample/sample_prices.csv",
    "debt": "data/sample/sample_debt.csv",
    "fundamentals": "data/sample/sample_fundamentals.csv",
    "rf": "data/sample/sample_risk_free.csv"}

parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true",
                    help="run on the synthetic sample data instead of the licensed panels")
args = parser.parse_args()

paths = sample_paths if args.sample else raw_paths

if args.sample:
    print("Running on synthetic sample data.")


# import all relevant data sets
prices_data = pd.read_csv(paths["prices"])
debt_data = pd.read_csv(paths["debt"])
TUP_fundamentals = pd.read_csv(paths["fundamentals"]).dropna()

# annual averages of daily 10 Year Treasury Yields
if args.sample:
    rf_data = pd.read_csv(paths["rf"], index_col=0, parse_dates=True)
else:
    rf_data = web.DataReader("RIFLGFCY10NA", "fred", "2013-01-01", "2022-12-31")


#====================================================================================
# Adjust dataframes to include all necessary columns for calculaions
# adjusted prices and adjusted outstanding shares
# PRC - Daily Close Price from CRSP
# CFACPR - Cumulative factor to adjust prices for stock splits from CRSP
# SHROUT - number of outstanding shares from CRSP
# CFACSHR - Cumulative factor to adjust outstanding shares for stock splits from CRSP
prices_data["adj_price"] = prices_data["PRC"] / prices_data["CFACPR"]
prices_data["adj_shrout"] = prices_data["SHROUT"] * prices_data["CFACSHR"]

# daily market cap using adjusted prices and shares outstanding in $M
prices_data["MC"] = prices_data["adj_price"] * prices_data["adj_shrout"] / 1000

# daily adjusted returns
# RET - Daily return adjusted for dividends from CRSP
# DLRET - delisting return from CRSP
prices_data["ret"] = (1 + prices_data["RET"]) * (1 + prices_data["DLRET"].fillna(0)) - 1
prices_data["log_ret"] = np.log(1 + prices_data["ret"])

# convert dates to datetime object
prices_data["date"] = pd.to_datetime(prices_data["date"])
debt_data["datadate"] = pd.to_datetime(debt_data["datadate"])
TUP_fundamentals["datadate"] = pd.to_datetime(TUP_fundamentals["datadate"])


#====================================================================================
""" 
    Compute one year default probabilities based on
    Merton Model
    Barrier Option Approach
    Jump Diffusion Process
"""
prices_df = prices_data.sort_values("date").copy()
debt_df = debt_data.sort_values("datadate").copy()

results_merton = []
results_barrier = []
results_jump = []
results_combined = []

years = sorted(prices_df["date"].dt.year.unique())[:-1]

for y in years:
    df_y = prices_df[prices_df["date"].dt.year == y]

    end_date = df_y["date"].max()
    prices_one_year = prices_df[prices_df["date"] <= end_date].tail(252)

    # Default Barrier
    lagged_date = end_date - pd.Timedelta(days=90)
    D = MertonModel.get_D(debt_df, lagged_date)

    rf_y = (rf_data.loc[:end_date, "RIFLGFCY10NA"].iloc[-1]) / 100.0
    # Infer assets and asset vol
    V, mu, sigmaV, sigmaE = MertonModel.merton(prices_one_year, D, rf_y)

    V0 = V[-1] # inital value based on merton model
    # for every year estimate PD 
    DD, PD = MertonModel.pd_one_yr(V0, D, mu, sigmaV)
    PD_barrier = BlackCoxBarrier.barrier_option(V0, D, mu, sigmaV)
    PD_jump, defaults_yr, n_sims_yr = JumpDiffusion.simulate_pd_jump_diffusion_barrier(V0=V0, D=D, mu=mu, sigma=sigmaV)

    results_merton.append({
        "year": y + 1,
        "D": D,
        "sigmaE": sigmaE,
        "sigmaV": sigmaV,
        "mu": mu,
        "DD_Merton": DD,
        "PD_Merton": PD})

    results_barrier.append({
        "year": y + 1,
        "D": D,
        "V0": V0,
        "sigmaV": sigmaV,
        "mu": mu,
        "PD_barrier": PD_barrier})
    
    results_jump.append({
        "year": y + 1,
        "number_of_defaults": defaults_yr,
        "number of simulations": n_sims_yr,
        "PD_jump_barrier": PD_jump})
    
    results_combined.append({
        "year": y + 1,
        "PD (Merton)": PD,
        "PD (Black & Cox)": PD_barrier,
        "PD (Jump Diffusion)": PD_jump})

merton_table = pd.DataFrame(results_merton)
barrier_table = pd.DataFrame(results_barrier)
jump_table = pd.DataFrame(results_jump)
res_comb = pd.DataFrame(results_combined)

print(merton_table)
print(barrier_table)
print(jump_table)
print(res_comb)


#====================================================================================
# Plot one year defualt probability of all models
plt.figure(figsize=(10, 6))
plt.plot(merton_table["year"], merton_table["PD_Merton"], label="Merton Model")
plt.plot(merton_table["year"], barrier_table["PD_barrier"], label="Barrier Option")
plt.plot(jump_table["year"], jump_table["PD_jump_barrier"], label="Jump Diffusion")
plt.xlabel("Year", fontsize=12)
plt.xticks(rotation=45)
plt.ylabel("Probability of Default", fontsize=12)
plt.title("One Year Probability of Default for TUP", fontsize=14, fontweight="bold")
plt.legend()
plt.show()


#====================================================================================
# Run Merton Model for plot
V, mu, sigmaV, sigmaE = MertonModel.merton(prices_data, D, rf_y)

# default barrier using KMV approach
debt_data["D"] = debt_data["dlcq"] + 0.5 * debt_data["dlttq"]

# Plot V_t from Merton Model vs Default level
plt.figure(figsize=(10, 6))
plt.plot(prices_data["date"], V)
n_years = (len(debt_data) - 1) // 4
for i in range(n_years):
    plt.hlines(y=debt_data["D"][4*i + 4],
               xmin=debt_data["datadate"][4*i],
               xmax=debt_data["datadate"][4*i + 4],
               colors="black",
               linewidth=2)
plt.hlines(y=debt_data["D"].iloc[-1],
           xmin=debt_data["datadate"].iloc[4 * n_years],
           xmax=prices_data["date"].iloc[-1],
           colors="black",
           linewidth=2)
plt.ylabel("Market Value of TUP (in $M)", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.xticks(rotation=45)
plt.title("Market Value vs Default Barrier", fontsize=14, fontweight="bold")
plt.legend(["V", "D"])
plt.show()


#====================================================================================
# Compute Altman Z-Score
z_score = Altman.altman_z(TUP_fundamentals)

# Plot for Altman Z-Score
plt.figure(figsize=(10, 6))
plt.plot(TUP_fundamentals["datadate"], z_score, label="Z-Score")
plt.axhline(y=1.81, color="red", label="Default Threshold")
plt.axhline(y=2.99, color="green", label="Survival Threshold")
ax = plt.gca()
ticks = ax.get_xticks()
ax.set_xticks(ticks[::4])
plt.xlabel("Year", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Altman Z-Score for TUP", fontsize=14, fontweight="bold")
plt.legend()
plt.show()


#====================================================================================
# Plot Tupperware revenue
plt.figure(figsize=(10,6))
plt.plot(TUP_fundamentals["datadate"], TUP_fundamentals["revtq"])
plt.xticks(rotation=45)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Revenue ($M)", fontsize=12)
plt.title("Tupperware Brands Quarterly Revenue (2013-2024)", fontsize=14, fontweight="bold")
plt.show()


#====================================================================================
# How does changing lambda affect the jump diffusion process PD

lam = [1, 5, 10, 20]
years = sorted(prices_df["date"].dt.year.unique())[:-1]

rows = []

for y in years:
    df_y = prices_df[prices_df["date"].dt.year == y]
    end_date = df_y["date"].max()

    prices_one_year = prices_df[prices_df["date"] <= end_date].tail(252)
    
    lagged_date = end_date - pd.Timedelta(days=90)
    D = MertonModel.get_D(debt_df, lagged_date)
    rf_y = rf_data.loc[:end_date, "RIFLGFCY10NA"].iloc[-1] / 100.0

    V, mu, sigmaV, sigmaE = MertonModel.merton(prices_one_year, D, rf_y)
    
    V0 = V[0]

    for l in lam:
        PD_jump, defaults_yr, n_sims_yr = JumpDiffusion.simulate_pd_jump_diffusion_barrier(V0=V0, D=D, mu=mu, sigma=sigmaV, n_sims=500, lam=l)

        rows.append({"year": y, "lam": l, "PD": PD_jump})


df_dlambda = pd.DataFrame(rows)

results_dlambda = (df_dlambda
           .pivot(index="year", columns="lam", values="PD")
           .rename(columns=lambda l: f"lam{lam.index(l)+1}_PD")
           .reset_index())

print(results_dlambda)


plt.figure(figsize=(10,6))
plt.plot(results_dlambda["year"], results_dlambda["lam1_PD"])
plt.plot(results_dlambda["year"], results_dlambda["lam2_PD"])
plt.plot(results_dlambda["year"], results_dlambda["lam3_PD"])
plt.plot(results_dlambda["year"], results_dlambda["lam4_PD"])
plt.ylabel("Default Probability", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.xticks(rotation=45)
plt.title("Default Probability with varying lambda", fontsize=14, fontweight="bold")
plt.legend(["lambda = 1", "lambda = 5", "lambda = 10", "lambda = 20"])
plt.show()


#====================================================================================
# How does changing muJ affect the jump diffusion process PD

muJ_list = [-0.01, -0.05, -0.1, -0.15]
years = sorted(prices_df["date"].dt.year.unique())[:-1]

rows = []

for y in years:
    df_y = prices_df[prices_df["date"].dt.year == y]
    end_date = df_y["date"].max()

    prices_one_year = prices_df[prices_df["date"] <= end_date].tail(252)
    
    lagged_date = end_date - pd.Timedelta(days=90)
    D = MertonModel.get_D(debt_df, lagged_date)
    rf_y = rf_data.loc[:end_date, "RIFLGFCY10NA"].iloc[-1] / 100.0

    V, mu, sigmaV, sigmaE = MertonModel.merton(prices_one_year, D, rf_y)
    V0 = V[0]

    for m in muJ_list:
        PD_jump, defaults_yr, n_sims_yr = JumpDiffusion.simulate_pd_jump_diffusion_barrier(V0=V0, D=D, mu=mu, sigma=sigmaV, n_sims=500, muJ=m)

        rows.append({"year": y, "mu": m, "PD": PD_jump})

df_dmu = pd.DataFrame(rows)

results_dmu = (df_dmu
           .pivot(index="year", columns="mu", values="PD")
           .rename(columns=lambda l: f"mu{muJ_list.index(l)+1}_PD")     # mu1_PD, mu2_PD, ...
           .reset_index())

print(results_dmu)


plt.figure(figsize=(10,6))
plt.plot(results_dmu["year"], results_dmu["mu1_PD"])
plt.plot(results_dmu["year"], results_dmu["mu2_PD"])
plt.plot(results_dmu["year"], results_dmu["mu3_PD"])
plt.plot(results_dmu["year"], results_dmu["mu4_PD"])
plt.ylabel("Default Probability", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.xticks(rotation=45)
plt.title("Default Probability with varying mu", fontsize=14, fontweight="bold")
plt.legend(["mu = -0.01", "mu = -0.05", "mu = -0.1", "mu = -0.15"])
plt.show()
