import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pandas_datareader import data, wb
import cvxpy


class portfolio_optimization():

    def __init__(self, tickers, source, period):
        self.enddate = datetime.datetime.today()
        self.period = datetime.timedelta(days=np.floor(365 * period))
        self.startdate = datetime.datetime.today() - self.period
        self.tickers = tickers
        self.source = source
        self.port_data = self.dl_data(self.tickers, self.source, self.startdate, self.enddate)
        self.pctRet = self.port_returns(self.port_data)

    def dl_data(self, tickers, source, start, end):
        port = data.DataReader(tickers, source, start, end)
        portcl = port['Close']
        valid_tickers = portcl.keys()[((portcl.isnull().sum()) <= 100).values]
        portvalid = portcl[valid_tickers]
        portvalid = portvalid.dropna(axis=0)
        return portvalid


    def port_returns(self, port):
        pct_ret = port.pct_change()
        pct_ret = pct_ret.dropna()
        return pct_ret


    def port_alloc_rand(self, port_returns):
        npts = 100
        sigmas = []
        mus = []

        p = np.asmatrix(np.mean(port_returns, axis=0))
        C = np.asmatrix(np.cov(port_returns.T))

        for i in xrange(npts):
            w = np.random.random(port_returns.shape[1])
            w = np.asmatrix(w/sum(w))

            mus.append((w * p.T).getA1())
            sigmas.append((np.sqrt(w*C*w.T)).getA1())

        return sigmas, mus

    def port_opt_classic(self, port_returns):
        p = (np.mean(port_returns, axis=0)).values
        w = cvxpy.Variable(port_returns.shape[1])

        gamma = cvxpy.Parameter(sign= 'positive')
        total_ret = w.T * p.T #doule check the T
        sigma = np.cov(port_returns.T)
        risk = cvxpy.quad_form(w, sigma)
        prob = cvxpy.Problem(cvxpy.Maximize(total_ret - gamma*risk),
                             [cvxpy.sum_entries(w)==1, w >= 0])

        N = 40  #number of points for the curve
        gamma_vals = np.logspace(0.5, 2, num=N)
        risks = []
        returns = []
        allocs = []
        for i in range(N):
            gamma.value = gamma_vals[i]
            prob.solve()
            risks.append(cvxpy.sqrt(risk).value)
            returns.append(total_ret.value)
            allocs.append(w.value.getA1())

        return risks, returns, allocs


    def port_opt_leverage(self):
        p = np.asmatrix(np.mean(self.pctRet, axis=0))
        w = cvxpy.Variable(len(self.ticker))
        gamma = cvxpy.Parameter(sign='positive')
        totalRet = w.T * p.T

        sigma = np.cov(self.pctRet.T)
        risk = cvxpy.quad_form(w, sigma)

        Lmax = cvxpy.Parameter()
        L_vals = [1, 1.5, 2, 2.5]

        prob = cvxpy.Problem(cvxpy.Maximize(totalRet),
                             [cvxpy.sum_entries(w) == 1,
                              cvxpy.norm(w,1) <= Lmax,
                              risk <= 2])

        w_vals = []
        for k, L_val in enumerate(L_vals):
            Lmax.value = L_val
            prob.solve()
            w_vals.append(w.value)

        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for k, L_val in enumerate(L_vals):
                plt.bar(np.arange(4)+k*0.25-0.375, (np.array(w_vals[k])),
                        color = colors[k], width = 0.25)
        plt.show()

    def port_opt_factor(self):

        p = np.asmatrix(np.mean(self.pctRet, axis=0))
        w = cvxpy.Variable(len(self.ticker))
        gamma = cvxpy.Parameter(sign='positive')
        totalRet = w.T * p.T

        F = np.random.random(len(self.ticker))
        D = np.diag(np.random.uniform(0, 0.9, size=len(self.ticker)))
        f = w.T * F.T
        Lmax = cvxpy.Parameter()

        f_sim = np.random.random(len(self.pctRet))
        sigma_f = np.cov(f_sim.T)
        # risk = cvxpy.quad_form(f, sigma_f) + cvxpy.quad_form(w, D)
        #
        # prob_factor = cvxpy.Problem(cvxpy.Maximize(totalRet - gamma*risk),
        #                             [cvxpy.sum_entries(w) == 1,
        #                              cvxpy.norm(w, 1) <= Lmax])

        # Standard portfolio optimization with data from factor model.
        risk = cvxpy.quad_form(w, F.dot(sigma_f).dot(F.T) + D)
        prob_factor = cvxpy.Problem(cvxpy.Maximize(totalRet - gamma * risk),
                       [cvxpy.sum_entries(w) == 1,
                        cvxpy.norm(w, 1) <= Lmax])

        Lmax.value = 2
        gamma.value = 0.1
        prob_factor.solve(verbose = True)

def main():
    ticker = ["GOOG", "AAPL", "MSFT", "FB"]
    source = "yahoo"
    period = 0.5
    f = open('nasdaq100.txt', 'rb')
    NSDAQ100 = f.read().strip().split()


    PO = portfolio_optimization(ticker, source, period)

    # classic
    risks, returns, allocs = PO.port_opt_classic(PO.pctRet)
    plt.plot(risks, returns, 'r-s')


    # leverage
    # PO.port_opt_leverage()

    # factor
    # PO.port_opt_factor()

    # simulated with random w
    risks, returns = PO.port_alloc_rand(PO.pctRet)
    plt.plot(risks, returns, 'o', markersize=5)
    plt.xlabel('risks')
    plt.ylabel('returns')
    plt.show()

    return 0


if __name__=='__main__':
    main()