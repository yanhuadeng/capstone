
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

import numpy as np
import datetime
import pandas as pd
from pandas_datareader import data, wb
# import pickle

from PortfolioAnalysis import PortfolioAnalysis




class PCA():
    def __init__(self):
        # self.portclose = pickle.load(open('port.p', 'rb'))
        self.timerange = pd.date_range('2007-7-1', '2016-7-1')
        self.negeigvals_ratio = []


    def cal_negeigvals_ratio(self, port_data):

        for ti in self.timerange:
            tistr_end = ti.strftime('%Y%m%d')
            tistr_beg = (ti - pd.DateOffset(60)).strftime('%Y%m%d')
            port_temp = port_data[tistr_beg:tistr_end]
            port_norm = (port_temp - port_temp.mean())/port_temp.std()
            port_cov = np.cov(port_norm.T)
            self.eigvalues, self.eigvectors = np.linalg.eigh(port_cov)
            pca1 = self.eigvalues.argmax()

            self.negeigvals_ratio.append(float((self.eigvectors[:, pca1]<0).sum())/len(self.eigvectors))

        #eigv, tick, eigvect = zip(*sorted(zip(self.eigvalues, self.portclose.columns, self.eigvectors)))


    def plot_ratio(self):
        plt.plot(self.timerange, self.negeigvals_ratio)
        plt.show()


    def plot_PCA(self):
        plt.pcolor(self.eigvectors)
        plt.show()


def main():
    tickers = ['AAPL', 'AMAT', 'AMGN', 'CERN',
                             'CHKP', 'COST', 'CSCO', 'FAST', 'INTC', 'MSFT', 'MU',
                             'MYL', 'PCAR',  'SNDK', 'SYMC', 'WFM', 'XRAY', 'YHOO']
    source = 'yahoo'
    start = datetime.datetime(1998, 1, 1)
    end = datetime.datetime(2013, 12, 31)

    period = 20
    deltaInd = 0

    PA = PortfolioAnalysis(port_tickers = tickers, period=10)
    PCAt = PCA()

    PCAt.cal_negeigvals_ratio(PA.port_data)
    PCAt.plot_ratio()
    # PCAt.plot_PCA()


    return 0


if __name__=='__main__':
    main()