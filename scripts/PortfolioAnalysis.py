from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime

# import matplotlib
# matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
# import cvxpy


class PortfolioAnalysis():
    """
    Class used for a what if analysis of a portfolio.  One of the factors that
    influences the total value is the overall index of the market.  Linear
    regression of each component of the portfolio is performed and changes in
    portfolio values are driven regression coefficients time the chages in
    index.  The change in the portfolio is uniformly distributed over the
    number of days taken in consideration but code is flexible enough to
    allow for implementation of other methods.  We can also add dependence
    on other factors such as GDP, interest rates, etc.
    """

    def __init__(self, tickers, source, period, idx_delta):
        """
        constructor for the class;  we take into account a history of one year
        :param ticker: list of tickers in the portfolio; by convention the first
            ticker is the index
        :param source: data source (string format) (yahoo, google, fed, etc)
        """
        self.enddate = datetime.datetime.today()
        self.period = datetime.timedelta(days=np.floor(365 * period))
        self.startdate = datetime.datetime.today() - self.period
        self.tickers = tickers
        self.source = source
        self.port_data = self.dl_data(self.tickers, self.source, self.startdate, self.enddate)
        self.daily_ret = self.port_returns(self.port_data)

        self.indexticker = tickers[0]
        self.idx_delta = idx_delta
        self.reg_coefficients, self.ols_dict = self.compute_regression()
        self.startPtVal = 0
        self.endPtVal = 0
        self.allocs_default = np.ones([1, len(tickers)-1])/float(len(tickers)-1)


    def dl_data(self, tickers, source, start, end):
        port = data.DataReader(tickers, source, start, end)
        portcl = port['Close']
        valid_tickers = portcl.keys()[((portcl.isnull().sum()) <= 100).values]
        portvalid = portcl[valid_tickers]
        portvalid = portvalid.dropna(axis=0)
        return portvalid

    def port_returns(self, port):
        daily_ret = port.pct_change()
        daily_ret = daily_ret.dropna()
        return daily_ret


    def compute_regression(self):
        """
        Compute regression coefficients on stock names and index.
        Zero intercept is assumed.
        :return: dataframe with regression coefficients values
        """
        df = pd.DataFrame(index=["const", self.indexticker], columns=self.port_data.columns, dtype='float')

        olsDict = {}
        for ticker in self.daily_ret.columns[0:len(self.daily_ret.columns)-1]:
            x = self.daily_ret[self.indexticker]
            y = self.daily_ret[ticker]
            ols = sm.OLS(y, sm.add_constant(x)).fit()
            olsDict[ticker] = ols
            df[ticker] = ols.params
        df.drop(self.indexticker, axis=1, inplace=True)

        return df, olsDict


    def plot_regression(self):
        # visual verification of the regression analysis

        plt.figure(1)
        x = self.daily_ret[self.indexticker]
        Nplt = len(self.tickers)-1
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for n in range(Nplt):
            # plt.subplot(Nplt,1,n+1)
            plt.plot(x, self.daily_ret[self.daily_ret.columns[n]], colors[n]+'o', label=self.daily_ret.columns[n])
            plt.plot(x, self.ols_dict[self.daily_ret.columns[n]].fittedvalues, colors[n]+'--')
            plt.legend()
            plt.xlabel('factor')
            plt.ylabel('components')
        plt.show()

    def cal_prediction(self, allocs):
        """
        Graphical view of the portfolio value changes.  Changes in index are
        read from a slider at the bottom of the plot
        :return:
        """
        alloc_df = pd.DataFrame(allocs, columns = self.tickers[1:])

        pastyear = self.port_data.index
        portfolioPast = self.port_data.sum(axis=1)
        futureyear = pastyear+self.period

        totalChange = self.cal_port_delta(self.idx_delta, alloc_df) #Index is not changed
        brownianVals = self.brownian(totalChange)
        brownianSeries = pd.DataFrame(np.array(brownianVals[:len(futureyear)]), index=futureyear)
        portfolioPast = pd.DataFrame(portfolioPast)
        portfolioPast.columns = ["Returns"]
        brownianSeries.columns = ["Returns"]
        return portfolioPast, brownianSeries

    def cal_port_delta(self, deltaInd, alloc):
        """
        Computed the change of the portfolio value given a change in index
        :param deltaInd: change in index
        :return: change in portfolio value
        """
        totalChange = 0.0
        for tkr in self.reg_coefficients.columns:
            beta = self.reg_coefficients[tkr].loc[self.indexticker]
            alpha = self.reg_coefficients[tkr].loc['const']
            totalChange += alpha + alloc[tkr] * beta * deltaInd
        return totalChange


    def brownian(self, totalChange):
        hurstExponent = 0.2
        scaleFactor = 2 ** (2.0 * hurstExponent)
        numPts = len(self.port_data)

        self.startPtVal = self.port_data.sum(axis=1).values[-1]
        self.endPtVal = self.startPtVal * totalChange  + self.startPtVal

        brownian_vals = []
        variance = 5000

        def brownian_bridge(x0, y0, x1, y1, variance, scaleFactor, brownian_vals):
            if (x1 - x0) < 1:
                brownian_vals.append(y0)
                return
            xm = (x0 + x1) / 2.0
            ym = (y0 + y1) / 2.0
            delta = np.random.normal(0, np.sqrt(variance),1)
            brownian_bridge(x0, y0, xm, ym+delta, variance/scaleFactor, scaleFactor, brownian_vals)
            brownian_bridge(xm, ym+delta, x1, y1, variance/scaleFactor, scaleFactor, brownian_vals)

        brownian_bridge(0, self.startPtVal, numPts, self.endPtVal, variance, scaleFactor, brownian_vals)

        return brownian_vals

    # def port_opt_classic(self, port_returns):
    #     p = np.asmatrix(np.mean(port_returns, axis=0))
    #     w = cvxpy.Variable(port_returns.shape[1])
    #
    #     gamma = cvxpy.Parameter(sign='positive')
    #     total_ret = w.T * p.T  # doule check the T
    #     sigma = np.cov(port_returns.T)
    #     risk = cvxpy.quad_form(w, sigma)
    #     prob = cvxpy.Problem(cvxpy.Maximize(total_ret - gamma * risk),
    #                          [cvxpy.sum_entries(w) == 1, w >= 0])
    #
    #     N = 40  # number of points for the curve
    #     gamma_vals = np.logspace(0.5, 2, num=N)
    #     risks = []
    #     returns = []
    #     allocs = []
    #     for i in range(N):
    #         gamma.value = gamma_vals[i]
    #         prob.solve()
    #         risks.append(cvxpy.sqrt(risk).value)
    #         returns.append(total_ret.value)
    #         allocs.append(w.value.getA1())
    #
    #     return risks, returns, allocs

    def port_alloc_rand(self, port_returns):
        npts = 200
        sigmas = []
        mus = []

        p = np.asmatrix(np.mean(port_returns, axis=0))
        C = np.asmatrix(np.cov(port_returns.T))

        for i in xrange(npts):
            w = np.random.random(port_returns.shape[1])
            w = np.asmatrix(w / sum(w))

            mus.append(np.array((w * p.T))[0][0])
            sigmas.append(np.array((np.sqrt(w * C * w.T)))[0][0])

        return sigmas, mus

def main():
    index = ['^GSPC']
    tickers = ['GOOG', 'AAPL', 'MSFT', 'FB']
    tickers.insert(0, index[0])

    f = open('nasdaq100.txt', 'rb')
    NASDAQ100 = f.read().strip().split()
    NASDAQ100.insert(0, index[0])

    source = "yahoo"

    period = 0.5
    deltaInd = .01

    PA = PortfolioAnalysis(tickers, source, period, deltaInd)

    risks, returns, allocs = PA.port_opt_classic(PA.daily_ret.ix[:, 0:len(tickers)-1])
    # plt.plot(risks, returns, 'r-s')

    # simulated with random w
    risks_rand, returns_rand = PA.port_alloc_rand(PA.daily_ret.ix[:, 0:len(tickers)-1])
    plt.plot(risks_rand, returns_rand, 'o', markersize=5)
    plt.xlabel('risks')
    plt.ylabel('returns')
    plt.show()

    past, future = PA.cal_prediction(allocs[0].reshape(1,len(allocs[0])))
    plt.plot(future['Returns'])
    plt.show()

    # PA.plot_regression()

if __name__ == "__main__":
    main()