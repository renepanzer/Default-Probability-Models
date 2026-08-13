import numpy as np


class Altman:

    @staticmethod
    # function to calculate all the ratios from df and return the Altman Z-Score based on Altman (1968)
    def altman_z(df_fundamentals):
        """ 
            The function asssumes that the dataframe contains the following columns (Data from Compustat and SEC filings):
            wcapq - Working Capital
            req - Retained Earnings
            mkvaltq - Market Cap
            revtq - Sales
            atq - Total Assets
            OperatingInc - operating income
            liabilities - total liabilites as reported in the balance sheet
            all data is on a quarterly basis.
        """
        x1 = np.array(df_fundamentals["wcapq"] / df_fundamentals["atq"])
        x2 = np.array(df_fundamentals["req"] / df_fundamentals["atq"])
        x3 = np.array(df_fundamentals["OperatingInc"].rolling(4).sum() / df_fundamentals["atq"])
        x4 = np.array(df_fundamentals["mkvaltq"] / df_fundamentals["liabilities"])
        x5 = np.array(df_fundamentals["revtq"].rolling(4).sum() / df_fundamentals["atq"])
        score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
        return score

