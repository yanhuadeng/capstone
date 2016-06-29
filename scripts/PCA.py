import pandas as pd
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import datetime
from pandas_datareader import data, wb
import pickle
import numpy as np

class PCA():
    def __init__(self):
        # self.portclose = pickle.load(open('port.p', 'rb'))
        self.ticker = tickers = ['AAPL', 'ALTR', 'AMAT', 'AMGN', 'CERN',
                    'CHKP', 'COST', 'CSCO', 'DELL', 'FAST', 'INTC', 'MSFT', 'MU',
                    'MYL', 'PCAR', 'SIAL', 'SNDK', 'SYMC', 'WFM', 'XRAY', 'YHOO']
        self.source = 'yahoo'
        self.start = datetime.datetime(1998, 1, 1)
        self.end = datetime.datetime(2013, 12, 31)
        self.portclose = self.download_data()
        self.timerange = pd.date_range('2006-7-1', '2010-7-1')
        self.negeigvals_ratio = []


    def download_data(self, tickers, source, start, end):
        # f = open('/Users/yhd/Google Drive/pycharm-projects/coding-exercises/nasdaq100.txt', 'r')
        # tickers = f.read().strip().split()

        port = data.DataReader(tickers, 'yahoo', start, end)
        portcl = port['Close']
        valid_tickers = portcl.keys()[((portcl.isnull().sum()) <= 100).values]
        portvalid = portcl[valid_tickers]
        portvalid = portvalid.dropna(axis=0)

        # pickle.dump(portvalid, open('port.p', 'wb'))

        return portvalid





    def cal_negeigvals_ratio(self):
        for ti in self.timerange:
            tistr_end = ti.strftime('%Y%m%d')
            tistr_beg = (ti - pd.DateOffset(60)).strftime('%Y%m%d')
            port_temp = self.portclose[tistr_beg:tistr_end]
            port_norm = (port_temp - port_temp.mean())/port_temp.std()
            portcov = np.cov(port_norm.T)
            self.eigvalues, self.eigvectors = np.linalg.eigh(portcov)
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
    pca_analysis = PCA()
    pca_analysis.cal_negeigvals_ratio()
    # pca_analysis.plot_ratio()
    pca_analysis.plot_PCA()

    return 0


if __name__=='__main__':
    main()