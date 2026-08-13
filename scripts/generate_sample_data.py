#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic panels with the same schema as the licensed inputs.
The sample runs 2013-2022,instead of 2013-2024 study period, because
FY2022 is the last 10K Tupperware filed. These panels exist only so the code
can be run without WRDS access - they are not intended to reproduce the results.
Sources for the anchors (from SEC filings):
    FY2013 10K
    FY2022 10K
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web


class Anchors:
    """
    Reported figures from the SEC filings, all in $M unless stated.
    """

    # FY2013 10-K 
    assets_2013 = 1843.9    # total assets
    current_debt_2013 = 235.4   # short-term borrowings + current portion of long-term debt
    lt_debt_2013 = 619.9    # long-term obligations
    equity_2013 = 252.9     # shareholder equity
    retained_earnings_2013 = 1289.2     # Retained earnings 
    working_capital_2013 = 41.5     # Working Capital
    revenue_2013 = 2671.6   # net sales, annual
    EBIT_2013 = 403.5   # operating income, annual
    market_cap_2013 = 4013.9    # aggregate market value of common equity, 28 Jun 2013
    shares_2013 = 50358.3   # shares outstanding in thousands, 20 Feb 2014
    price_2013 = 1000 * market_cap_2013 / shares_2013  # approximate $ per share at 28 Jun 2013
    

    # FY2022 10-K 
    assets_2022 = 743.6
    current_debt_2022 = 709.8
    lt_debt_2022 = 3.1
    equity_2022 = -429.8
    retained_earnings_2022 = 887.3    
    working_capital_2022 = -41.5    # figure is made up
    revenue_2022 = 1304.0
    EBIT_2022 = 23.6
    market_cap_2022 = 305.6    # aggregate market value of common equity, 24 Jun 2022
    shares_2022 = 46269.3   # shares outstanding in thousands, 20 Feb 2023
    price_2022 = 1000 * market_cap_2022 / shares_2022  # approximate $ per share at 28 Jun 2023
    

    # Total liabilities = total assets - equity
    liabilities_2013 = assets_2013 - equity_2013
    liabilities_2022 = assets_2022 - equity_2022



class SyntheticFirm:
    
    # Trading days per year
    trading_days = 252


    @staticmethod
    def interpolate(start, end, n):
        """
        Linear interpolation
        Input parameters:
            start = first value of the sample
            end  = last value of the sample
            n = number of observations in the sample
        """
        return np.linspace(start, end, n)


    @staticmethod
    def daily_prices(start_date, end_date, seed=42):
        """
        Synthetic daily price series. 
        Input parameters:
            start_date = first date of the sample
            end_date = last date of the sample
            seed = random seed for reproducibility
        """
        np.random.seed(seed)

        dates = pd.bdate_range(start_date, end_date)
        n = len(dates)

        # target total log return over the sample, implied by the price anchors
        total_log_return = np.log(Anchors.price_2022 / Anchors.price_2013)

        # single random volatility level for the whole sample
        sigma_annual = np.random.uniform(0.25, 0.60)
        sigma_daily = sigma_annual / np.sqrt(252)

        # demean the shocks first, so the realised path lands on the end anchor
        # rather than only doing so in expectation
        shocks = sigma_daily * np.random.randn(n)
        shocks = shocks - shocks.mean()

        drift = total_log_return / n
        log_ret = drift + shocks

        price = Anchors.price_2013 * np.exp(np.cumsum(log_ret))

        # shares outstanding held flat
        shrout = np.full(n, Anchors.shares_2013)

        prices_df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "PRC": np.round(price, 2),
            "SHROUT": shrout,
            "CFACPR": 1.0,
            "CFACSHR": 1.0,
            "RET": np.round(np.expm1(log_ret), 6),
            "DLRET": np.nan})

        return prices_df


    @staticmethod
    def quarterly_debt(start_date, end_date):
        """
        Build synthetic quarterly debt panel.
        """
        quarters = pd.date_range(start_date, end_date, freq="QE")
        n = len(quarters)

        # short-term debt
        st_debt = SyntheticFirm.interpolate(Anchors.current_debt_2013, Anchors.current_debt_2022, n)

        # long-term debt
        lt_debt = SyntheticFirm.interpolate(Anchors.lt_debt_2013, Anchors.lt_debt_2022, n)

        debt_df = pd.DataFrame({
            "datadate": quarters.strftime("%Y-%m-%d"),
            "dlcq": np.round(st_debt, 2),
            "dlttq": np.round(lt_debt, 2)})

        return debt_df


    @staticmethod
    def quarterly_fundamentals(start_date, end_date, debt_df, seed=42):
        """
        Build synthetic quarterly fundamentals panel.
        Input parameters:
            debt_df = output of quarterly_debt
            seed = random seed for noise 
        """
        np.random.seed(seed + 1)

        quarters = pd.date_range(start_date, end_date, freq="QE")
        n = len(quarters)

        # Total assets
        assets = SyntheticFirm.interpolate(Anchors.assets_2013, Anchors.assets_2022, n)

        # Total liabilities
        liabilities = SyntheticFirm.interpolate(Anchors.liabilities_2013, Anchors.liabilities_2022, n)

        # quarterly flows, with added noise so the trailing sums are not perfectly linear
        revenue = SyntheticFirm.interpolate(Anchors.revenue_2013, Anchors.revenue_2022, n) / 4.0
        revenue = revenue * np.exp(0.1 * np.random.randn(n))

        op_income = SyntheticFirm.interpolate(Anchors.EBIT_2013, Anchors.EBIT_2022, n) / 4.0
        op_income = op_income + 4.0 * np.random.randn(n)

        # market cap
        mktcap = mktcap = np.exp(SyntheticFirm.interpolate(np.log(Anchors.market_cap_2013), np.log(Anchors.market_cap_2022), n))

        # working capital turns negative from 2013 reported figure
        wcap = SyntheticFirm.interpolate(Anchors.working_capital_2013, Anchors.working_capital_2022, n)

        # Retained earnings
        retained = SyntheticFirm.interpolate(Anchors.retained_earnings_2013, Anchors.retained_earnings_2022, n)

        fundamentals_df = pd.DataFrame({
            "datadate": quarters.strftime("%Y-%m-%d"),
            "atq": np.round(assets, 2),
            "liabilities": np.round(liabilities, 2),
            "wcapq": np.round(wcap, 2),
            "req": np.round(retained, 2),
            "OperatingInc": np.round(op_income, 2),
            "revtq": np.round(revenue, 2),
            "mkvaltq": np.round(mktcap, 2),
            "dlcq": debt_df["dlcq"].values,
            "dlttq": debt_df["dlttq"].values})

        return fundamentals_df


    @staticmethod
    def annual_risk_free(start_date, end_date):
        """
        Annualised average 10-year Treasury yield from FRED, in %.
        """
        rf_df = web.DataReader("RIFLGFCY10NA", "fred", start_date, end_date)
        rf_df.index.name = "DATE"

        return rf_df


#====================================================================================
# Generate all panels and write them to data/sample

outdir = "data/sample"
start_date = "2013-01-01"
end_date = "2022-12-31"
seed = 42

prices_data = SyntheticFirm.daily_prices(start_date, end_date, seed)
debt_data = SyntheticFirm.quarterly_debt(start_date, end_date)
fundamentals_data = SyntheticFirm.quarterly_fundamentals(start_date, end_date, debt_data, seed)
rf_data = SyntheticFirm.annual_risk_free(start_date, end_date)

prices_data.to_csv(f"{outdir}/sample_prices.csv", index=False)
debt_data.to_csv(f"{outdir}/sample_debt.csv", index=False)
fundamentals_data.to_csv(f"{outdir}/sample_fundamentals.csv", index=False)
rf_data.to_csv(f"{outdir}/sample_risk_free.csv")

print(f"Wrote synthetic panels to {outdir}")
print(f"prices: {len(prices_data)} daily rows")
print(f"debt: {len(debt_data)} quarterly rows")
print(f"fundamentals:{len(fundamentals_data)} quarterly rows")
print(f"risk free: {len(rf_data)}annual rows")

