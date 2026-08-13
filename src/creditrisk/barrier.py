import numpy as np
from scipy.stats import norm


class BlackCoxBarrier:

    # Probability of default using Black Cox Barrier option approach
    @staticmethod
    def barrier_option(V0, D, mu, sigma, T=1):
        # if market value < default threshold => firm defaulted
        if V0 <= D:
            return 1.0

        r = mu - 0.5 * sigma**2
        z1 = (np.log(D/V0) - r*T) / (sigma * T**0.5)
        z2 = (np.log(D/V0) + r*T) / (sigma * T**0.5)
        PD = norm.cdf(z1) + np.exp(2*r*np.log(D/V0) / sigma**2) * norm.cdf(z2)
        return PD
