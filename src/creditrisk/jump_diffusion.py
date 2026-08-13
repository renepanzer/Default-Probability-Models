import numpy as np


class JumpDiffusion:

    @staticmethod
    def simulate_pd_jump_diffusion_barrier(V0, D, mu, sigma, steps=252, n_sims=5000,
                                           lam=12, muJ=-0.05, sigmaJ=0.1, seed=1):
        """
            Input parameters (based on Pollastri et al.):
            V0 - Initial asset value
            D - Default barrier 
            steps - 252 days per year
            n_sims - number of simulations
            mu - Annual drift 
            sigma - Annualized volatility of asset
            lam - Jump intensity 
            muJ - Mean of log jump size
            sigmaJ - Standard Deviation jump size
        """
        T = 1 # time horizon
        np.random.seed(seed)
        # time step based on function inputs
        dt = T / steps
        # Jump compensator k = E[J - 1]
        EJ = np.exp(muJ + 0.5 * sigmaJ**2)
        k = EJ - 1.0
        # Drift adjusted for jumps
        mu_adj = mu - lam * k

        defaults = 0

        for sim in range(n_sims):
            # intialize V
            V = V0
            defaulted = False

            for i in range(steps):
                # diffusion step
                Z = np.random.randn()
                V = V * np.exp((mu_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

                # jumps in this small interval
                Nj = np.random.poisson(lam * dt)

                if Nj > 0:
                    # apply Nj lognormal jumps multiplicatively
                    # product of exp(Y) = exp(sum(Y))
                    Ysum = np.random.normal(muJ, sigmaJ, size=Nj).sum()
                    V = V * np.exp(Ysum)

                # if Market Value hits the default barrier, break loop and add 1 to default counter
                if V <= D:
                    defaulted = True
                    break

            if defaulted:
                defaults += 1
                
        # compute fraction of defaults in the number of simulations
        PD_sim = defaults / n_sims
        
        return PD_sim, defaults, n_sims
