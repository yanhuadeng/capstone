import pandas.io.data as web
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt


class FinanceWhatIf():
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

    def __init__(self, ticker, source, period, deltaInd):
        """
        constructor for the class;  we take into account a history of one year
        :param ticker: list of tickers in the portfolio; by convention the first
            ticker is the index
        :param source: data source (string format) (yahoo, google, fed, etc)
        """
        self.enddate = datetime.datetime.today()
        # one year history
        self.period = datetime.timedelta(days=np.floor(365*period))
        self.startdate = datetime.datetime.today() - self.period
        self.indexticker = ticker[0]
        self.ticker = ticker
        self.source = source
        self.deltaInd = deltaInd
        self.data = self.getFinancialData()
        self.regCoeff, self.olsDict = self.computeRegression()
        self.startPtVal = 0
        self.endPtVal = 0
        # self.portfolioPlot()


    def getFinancialData(self):
        """
        Get time series with financial data
        :return: by convention a dataframe with the close values, and index the
        number of days
        """
        data = web.DataReader(self.ticker, self.source, self.startdate, self.enddate)
        close = data["Close"]
        return close


    def computeRegression(self):
        """
        Compute regression coefficients on stock names and index.
        Zero intercept is assumed.
        :return: dataframe with regression coefficients values
        """
        df = pd.DataFrame(index=["const", self.indexticker], columns=self.data.columns, dtype='float')
        self.retDF = self.data.pct_change()
        self.retDF.dropna(inplace=True)
        olsDict = {}
        for ticker in self.retDF.columns[0:len(self.retDF.columns)-1]:
            x = self.retDF[self.indexticker]
            y = self.retDF[ticker]
            ols = sm.OLS(y, sm.add_constant(x)).fit()
            olsDict[ticker] = ols
            df[ticker] = ols.params
        df.drop(self.indexticker, axis=1, inplace=True)

        return df, olsDict

    # visual verification of the regression analysis

    def regressionPlots(self):
        plt.figure(1)
        x = self.retDF[self.indexticker]
        Nplt = len(self.ticker)-1
        for n in range(Nplt):
            plt.subplot(Nplt,1,n+1)
            plt.plot(x, self.retDF[self.retDF.columns[n]], 'o', label="data")
            plt.plot(x, self.olsDict[self.retDF.columns[n]].fittedvalues, 'r--.')
            # plt.title = self.retDF.columns[n]
        plt.show()

    def portfolioPlot(self):
        """
        Graphical view of the portfolio value changes.  Changes in index are
        read from a slider at the bottom of the plot
        :return:
        """
        pastyear = self.data.index
        portfolioPast = self.data.sum(axis=1)
        futureyear = pastyear+self.period

        totalChange = self.deltaPortf(self.deltaInd) #Index is not changed
        brownianVals = self.BB(totalChange)
        brownianSeries = pd.DataFrame(np.array(brownianVals[:len(futureyear)]), index=futureyear)
        portfolioPast = pd.DataFrame(portfolioPast)
        portfolioPast.columns = ["Returns"]
        brownianSeries.columns = ["Returns"]
        return portfolioPast, brownianSeries

    def deltaPortf(self, deltaInd):
        """
        Computed the change of the portfolio value given a change in index
        :param deltaInd: change in index
        :return: change in portfolio value
        """
        totalChange = 0.0
        for tkr in self.regCoeff.columns:
            beta = self.regCoeff[tkr].loc[self.indexticker]
            alpha = self.regCoeff[tkr].loc['const']
            totalChange += alpha + beta * deltaInd
        return totalChange


    def BB(self, totalChange):
        hurstExponent = 0.2
        scaleFactor = 2 ** (2.0 * hurstExponent)
        numPts = len(self.data)

        self.startPtVal = self.data.sum(axis=1).values[-1]
        self.endPtVal = self.startPtVal * totalChange  + self.startPtVal

        brownianVals = []
        variance = 5000

        def BrownianBridge(x0, y0, x1, y1, variance, scaleFactor, brownianVals):
            if (x1 - x0) < 1:
                brownianVals.append(y0)
                return
            xm = (x0 + x1) / 2.0
            ym = (y0 + y1) / 2.0
            delta = np.random.normal(0, np.sqrt(variance),1)
            BrownianBridge(x0, y0, xm, ym+delta, variance/scaleFactor, scaleFactor, brownianVals)
            BrownianBridge(xm, ym+delta, x1, y1, variance/scaleFactor, scaleFactor, brownianVals)

        BrownianBridge(0, self.startPtVal, numPts, self.endPtVal, variance, scaleFactor, brownianVals)

        # brownianVals = brownianVals.append(endPtVal)
        return brownianVals

def main():
    index = ["^GSPC"]
    ticker = ["GOOG", "AAPL", "MSFT", "FB"]

    f = open('nasdaq100.txt', 'rb')
    NSDAQ100 = f.read().strip().split()

    NASDAQ100 = NSDAQ100.insert(0, index[0])
    source = "yahoo"
    period = 3
    deltaInd = .001
    N=4
    WT = FinanceWhatIf(NSDAQ100[0:N], source, period,deltaInd)
    WT.portfolioPlot()
    # WT.regressionPlots()

if __name__ == "__main__":
    main()