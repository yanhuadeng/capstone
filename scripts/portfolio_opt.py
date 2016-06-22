import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import datetime
import pandas.io.data as web


class portfolio_optimization():

    def __init__(self, ticker, source, period):
        self.enddate = datetime.datetime.today()
        self.period = datetime.timedelta(days=np.floor(365 * period))
        self.startdate = datetime.datetime.today() - self.period
        self.ticker = ticker
        self.source = source
        self.pctRet = self.getData()


    def getData(self):
        data = web.DataReader(self.ticker, self.source, self.startdate, self.enddate)
        data = data["Close"]
        data = data.dropna()
        pctRet = data.pct_change()
        pctRet = pctRet.dropna()
        return pctRet

    def randDist(self):
        k = np.random.random(len(self.ticker))
        return k/sum(k)

    def randPortDist(self):
        p = np.asmatrix(np.mean(self.pctRet, axis=0))
        w = np.asmatrix(self.randDist())
        C = np.asmatrix(np.cov(self.pctRet.T))

        mu = w * p.T
        sigma = np.sqrt(w*C*w.T)

        if sigma > 3:
            return randPortDist(self.pctRet)
        return mu, sigma

    def optPort(self):
        returns = self.pctRet
        n = len(returns)
        returns = np.asmatrix(returns)

        N = 10
        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        S = opt.matrix(np.cov(returns))
        pbar = opt.matrix(np.mean(returns, axis=1))

        # Create constraint matrices
        G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                      for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return np.asarray(wt), returns, risks


def main():
    ticker = ["GOOG", "AAPL", "MSFT", "FB"]
    source = "yahoo"
    period = 3
    f = open('nasdaq100.txt', 'rb')
    NSDAQ100 = f.read().strip().split()


    PO = portfolio_optimization(ticker, source, period)

    ndistr = 500

    means, stds = np.column_stack([PO.randPortDist()
                                   for _ in xrange(ndistr)])

    weights, returns, risks = PO.optPort()

    plt.plot(stds, means, 'o', markersize=5)

    plt.xlabel('std')
    plt.ylabel('mean')
    plt.plot(risks, returns, 'r-o')
    plt.show()

    return 0


if __name__=='__main__':
    main()