from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime

# import matplotlib
# matplotlib.use('tkAgg')
# import matplotlib.pyplot as plt

import statsmodels.api as sm
# import cvxpy
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import brewer

class PortfolioAnalysis():
    """
    Class used for a what if analysis of a portfolio.  One of the factors that
    influences the total value is the overall index of the market.  Linear
    regression of each component of the portfolio is performed and changes in
    portfolio values are driven regression coefficients time the changes in
    index.  The change in the portfolio is uniformly distributed over the
    number of days taken in consideration but code is flexible enough to
    allow for implementation of other methods.  We can also add dependence
    on other factors such as GDP, interest rates, etc.
    """

    def __init__(self, port_tickers, source = 'yahoo', period = 1, idx_ticker = []):
        """
        constructor for the class;  we take into account a history of one year
        :param ticker: list of tickers in the portfolio; by convention the first
            ticker is the index
        :param source: data source (string format) (yahoo, google, fed, etc)
        """
        self.enddate = datetime.datetime.today()
        self.period = datetime.timedelta(days=np.floor(365 * period))
        self.startdate = datetime.datetime.today() - self.period
        self.tickers = port_tickers
        self.indexticker = idx_ticker
        self.source = source
        self.tickersidx = port_tickers + idx_ticker
        self.startPtVal = 0
        self.endPtVal = 0


    def dl_data(self, tickers, source, start, end):
        port = data.DataReader(tickers, source, start, end)
        portcl = port['Close']
        valid_tickers = portcl.keys()[((portcl.isnull().sum()) <= 100).values]
        portvalid = portcl[valid_tickers]
        portvalid = portvalid.dropna(axis=0)
        self.valid_data = portvalid
        self.port_data = self.valid_data[self.tickers]
        self.idx_data = self.valid_data[self.indexticker]
        self.daily_ret = self.port_returns(self.valid_data)
        self.port_daily_ret = self.daily_ret[self.tickers]
        self.idx_daily_ret = self.daily_ret[self.indexticker]


        # return portvalid

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
        # df = pd.DataFrame(index=("const", self.indexticker[0]), columns=self.tickers, dtype='float')

        olsDict = {}
        # for ticker in self.tickers:
        x = self.idx_daily_ret
        y = self.port_daily_ret
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        olsDict = ols
        df = ols.params
        df.columns = self.tickers
        # df.drop(self.indexticker, axis=1, inplace=True)
        self.reg_coefficients = df
        self.ols_dict = olsDict


    def plot_regression(self):
        # visual verification of the regression analysis

        plt.figure(1)
        x = self.idx_daily_ret
        Nplt = len(self.tickers)
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for n in range(Nplt):
            # plt.subplot(Nplt,1,n+1)
            plt.plot(x, self.daily_ret[self.tickers[n]], colors[n]+'o', label=self.tickers[n])
            plt.plot(x, self.ols_dict[self.tickers[n]].fittedvalues, colors[n]+'--')
            plt.legend()
            plt.xlabel('factor')
            plt.ylabel('components')
        plt.show()

    def cal_prediction(self, deltaInd = 0, allocs = [1]):
        """
        Graphical view of the portfolio value changes.  Changes in index are
        read from a slider at the bottom of the plot
        :return:
        """
        if len(allocs) == 1:
            allocs = np.ones([1, len(self.tickers)]) / float(len(self.tickers))
        alloc_df = pd.DataFrame(allocs, columns = self.tickers)

        pastyear = self.port_data.index
        port_past = self.port_data.sum(axis=1)

        futureyear = pastyear+self.period

        totalChange = self.cal_port_delta(deltaInd, alloc_df) #Index is not changed
        brownian_vals = self.brownian(totalChange)
        brownian_series = pd.DataFrame(np.array(brownian_vals[:len(futureyear)]), index=futureyear)
        port_past = pd.DataFrame(port_past)
        port_past.columns = ["Returns"]
        brownian_series.columns = ["Returns"]
        return port_past, brownian_series

    def cal_port_delta(self, deltaInd, allocs):
        """
        Computed the change of the portfolio value given a change in index
        :param deltaInd: change in index
        :return: change in portfolio value
        """
        totalChange = 0.0
        for tkr in self.reg_coefficients.columns:
            beta = self.reg_coefficients[tkr].loc[self.indexticker]
            alpha = self.reg_coefficients[tkr].loc['const']
            totalChange += alpha + allocs[tkr].ix[0] * beta * deltaInd
        return totalChange


    def brownian(self, totalChange):
        hurstExponent = 0.1
        scaleFactor = 1 ** (2.0 * hurstExponent)
        numPts = len(self.port_data)


        self.startPtVal = self.port_data.sum(axis=1).values[-1]

        self.endPtVal = self.startPtVal * totalChange  + self.startPtVal

        brownian_vals = []
        variance = self.port_data.sum(axis=1).std()
        self.port_std = variance

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

    def port_opt_classic(self, port_returns):
        p = np.asmatrix(np.mean(port_returns, axis=0))
        w = cvxpy.Variable(port_returns.shape[1])

        gamma = cvxpy.Parameter(sign='positive')
        total_ret = w.T * p.T  # doule check the T
        sigma = np.cov(port_returns.T)
        risk = cvxpy.quad_form(w, sigma)
        prob = cvxpy.Problem(cvxpy.Maximize(total_ret - gamma * risk),
                             [cvxpy.sum_entries(w) == 1, w >= 0])

        N = 40  # number of points for the curve
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

    def port_alloc_rand(self, port_returns):
        npts = 200
        sigmas = []
        mus = []

        p = np.asmatrix(np.mean(port_returns, axis=0))
        C = np.asmatrix(np.cov(port_returns.T))
        weights = []
        for i in xrange(npts):
            w = np.random.random(port_returns.shape[1])
            w = np.asmatrix(w / sum(w))

            mus.append(np.array((w * p.T))[0][0])
            sigmas.append(np.array((np.sqrt(w * C * w.T)))[0][0])
            weights.append(np.array(w)[0])

        return sigmas, mus, weights

def main():
    index = ['^GSPC']
    tickers = ['GOOG', 'AAPL', 'MSFT', 'FB']
    # tickers.insert(0, index[0])

    # f = open('nasdaq100.txt', 'rb')
    # NASDAQ100 = f.read().strip().split()
    # NASDAQ100.insert(0, index[0])

    source = "yahoo"

    period = .5
    deltaInd = .01

    PA = PortfolioAnalysis(tickers, source, period, index)
    PA.dl_data(PA.tickersidx, PA.source, PA.startdate, PA.enddate)

    # risks, returns, allocs = PA.port_opt_classic(PA.port_daily_ret)
    # plt.plot(risks, returns, 'r-s')

    # simulated with random w
    risks_rand, returns_rand, weights = PA.port_alloc_rand(PA.port_daily_ret)
    # print risks_rand
    # print returns_rand
    # print np.argmax(returns_rand)

    # plt.plot(risks_rand, returns_rand, 'o', markersize=5)
    # plt.xlabel('risks')
    # plt.ylabel('returns')
    # plt.show()

    past, future = PA.cal_prediction(deltaInd = 0.01, allocs = allocs[0].reshape(1,len(allocs[0])))
    # plt.plot(past['Returns'], 'g')
    # plt.plot(future['Returns'], 'r')
    # plt.fill_between(future.index, future['Returns'][0]-PA.port_std, future['Returns'][0]+PA.port_std, alpha = 0.2, facecolor = 'r')
    # plt.show()

    # PA.plot_regression()
    # colors = brewer["Spectral"][len(tickers)]
    # fig_corr = figure()
    # for n in range(len(tickers)):
    #     fig_corr.scatter(PA.idx_daily_ret.values.T[0], PA.port_daily_ret.ix[:,n].values,
    #                      radius=.01*PA.port_daily_ret.max().max(), fill_color = colors[n], fill_alpha=0.2, legend=tickers[n])
    #     fig_corr.line(PA.idx_daily_ret.values.T[0], PA.ols_dict.fittedvalues.ix[:,n].values,
    #                   color = colors[n])  # , color = colors[n]
    #
    # output_file("brewer.html")
    # show(fig_corr)

if __name__ == "__main__":
    main()