import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class MertonModel:
    
    # Black-Scholes Call Option
    @staticmethod
    def bs_call(V, D, rf, sigmaV, T):
        """ 
            Input parameters:
            V = Market Value
            D = Default barrier
            rf = risk free rate
            sigmaV = Volatility estimate
            T = Time horizon
        """
        d1 = (np.log(V / D) + (rf + 0.5 * sigmaV**2) * T) / (sigmaV * np.sqrt(T))
        d2 = d1 - sigmaV * np.sqrt(T)

        return V * norm.cdf(d1) - D * np.exp(-rf * T) * norm.cdf(d2)


    # Difference equation BS call option - Stock Price at t
    @staticmethod
    def diff_call_minus_equity(V, S_t, D, rf, sigmaV, T):
        
        return MertonModel.bs_call(V, D, rf, sigmaV, T) - S_t


    # Find zero of the diﬀerence
    @staticmethod
    def solve_diff(S_t, D, rf, sigmaV, T):
        V_initial = S_t + D
        # root finder using fsolve
        V_sol = fsolve(
            MertonModel.diff_call_minus_equity,
            x0=V_initial,
            args=(S_t, D, rf, sigmaV, T))

        return V_sol[0]


    # annualised volatility from equity prices
    @staticmethod
    def sigma_annual(returns):
        # compute daily log returns
        sigma_annual = np.std(returns) * np.sqrt(252)
        
        return sigma_annual
    
    
    # Find the default threshold for every period
    @staticmethod
    def get_D(debt_df, end_date):
        temp = debt_df[debt_df["datadate"] <= end_date]
        row = temp.sort_values("datadate").iloc[-1]
        D = row["dlcq"] + 0.5 * row["dlttq"]
        return D


    # KMV Merton model
    @staticmethod
    def merton(prices, D, rf):
        # daily market cap from prices_data
        S_t = prices["MC"]
        rf = float(rf)
        returns = prices["log_ret"]
        # Time horizon = 1 year
        T = 1
        # estimate initial stock volatility based on historical log returns
        sigmaE = MertonModel.sigma_annual(returns)
        # initial market cap at t=0
        S0 = S_t.iloc[0]
        # estimate for sigmaV
        sigmaV = sigmaE * S0 / (S0 + D)
        
        diff = np.inf # tracks difference between old and new sigma; set to large initial value
        counter = 0 # counter to ensure algorithm converges or breaks
        while (diff > 0.0001) and (counter < 100):
            sigmaV_old = sigmaV # set old estimate for sigmaV to current sigmaV
            # Find daily Market Value using BS 
            V_t = []
            for S in S_t:
                V = MertonModel.solve_diff(S, D, rf, sigmaV, T)
                V_t.append(V)
            V_t = np.array(V_t)
            # estimate returns based on computed V_t
            asset_returns = np.log(V_t[1:] / V_t[:-1])
            
            # estimate new annualized sigmaV based on computed Market Values
            sigmaV = np.std(asset_returns) * np.sqrt(252)
            # difference between current and old volatility estimate
            # take absolute value to ensure algorithm doesn't converge immediately
            diff = abs(sigmaV - sigmaV_old)
            
            counter += 1
        # estimate mu as annualized past returns
        mu = asset_returns.mean() * 252

        return V_t, mu, sigmaV, sigmaE


    # 1-year Probability of Default and Distance to Default
    @staticmethod
    def pd_one_yr(V0, D, mu, sigmaV, T=1.0):
        DD = (np.log(V0 / D) + (mu - 0.5 * sigmaV**2) * T) / (sigmaV * np.sqrt(T))
        PD = norm.cdf(-DD)
        return DD, PD
