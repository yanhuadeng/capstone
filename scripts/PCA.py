import pandas as pd
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import datetime
from pandas_datareader import data, wb
import pickle
import numpy as np
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.palettes import brewer
from bokeh.charts import HeatMap, bins
from bokeh.models import PrintfTickFormatter

class PCA():
    def __init__(self):
        # self.portclose = pickle.load(open('port.p', 'rb'))
        self.offset_days = 60


    def dl_data(self, tickers, source, start, end):
        port = data.DataReader(tickers, source, start, end)
        port_cl = port['Close']
        if len(port_cl.shape) == 1:
            port_cl.columns = tickers
            port_valid = port_cl

        else:
            valid_tickers = port_cl.columns[((port_cl.isnull().sum()) <= 100).values]
            port_valid = port_cl[valid_tickers]

        port_valid = port_valid.dropna(axis=0)

        # daily_ret = port.pct_change()
        # daily_ret = daily_ret.dropna()
        # self.daily_ret = daily_ret

        return port_valid

    def cal_negeigvals_ratio(self, port_data):
        N = 60
        start = port_data.index[0]+datetime.timedelta(days=N)
        end = port_data.index[-1]
        timerange = pd.date_range(start, end)
        negeigvals_ratio = []
        for ti in timerange:
            tistr_end = ti.strftime('%Y%m%d')
            tistr_beg = (ti - pd.DateOffset(N)).strftime('%Y%m%d')
            port_60 = port_data[tistr_beg:tistr_end]
            port_norm = (port_60 - port_60.mean())/port_60.std()
            port_cov = np.cov(port_norm.T)
            eigvalues, eigvectors = np.linalg.eigh(port_cov)
            negeigvals_ratio.append(float((eigvectors[-1]<0).sum())/len(eigvectors))

        ratio_df = pd.DataFrame(negeigvals_ratio, index = timerange, columns = ['ratio'])
        return ratio_df, eigvectors

    def find_neg_ratio(self, port_data, ratio):
        ratio_dfavg = ratio['ratio'].rolling(window=30).mean()

        ratio_df = pd.DataFrame(np.zeros(len(ratio_dfavg)), index=ratio_dfavg.index, columns=['neg5'])
        ratio_df['port_data'] = port_data.sum(axis=1)
        ratio_df['ratio'] = ratio_dfavg
        ratio_df['ratio_n'] = ratio

        for i in xrange(5, len(ratio_df)):
            if (ratio_df['ratio'][i - 0] > ratio_df['ratio'][i - 1]
                and ratio_df['ratio'][i - 1] > ratio_df['ratio'][i - 2]
                and ratio_df['ratio'][i - 2] > ratio_df['ratio'][i - 3]):
                    ratio_df['neg5'][i] = ratio_df['port_data'][i]
        ratio_df = ratio_df.replace(0, np.nan)
        return ratio_df

    def pca_3(self):
        self.eigvalues3 = self.eigvalues[-3:]
        self.eigvectors3 = self.eigvectors[-3:]
        PCretDict = {}
        for i in range(len(self.eigvalues3)):
            key = 'PC' + str(i)
            PCret = np.sum(self.eigvectors3[i] * port_rets, axis=1)
            PCretDict[key] = PCret
        PCret = pd.DataFrame.from_dict(PCretDict)



    def plot_ratio(self):
        plt.plot(self.timerange, self.negeigvals_ratio)
        plt.show()


    def plot_PCA(self):
        plt.pcolor(self.eigvectors)
        plt.show()

    # TO DO
    # test case: if one of the components has number of shares much larger than
    # the rest then PC with largest eigenvalue should be that stock
    # all 1 share but AAPL has 1000 shares, PC1 should be AAPL


def main():
    tickers = ['AAPL', 'ALTR', 'AMAT', 'AMGN', 'CERN',
                 'CHKP', 'COST', 'CSCO', 'DELL', 'FAST', 'INTC', 'MSFT', 'MU',
                 'MYL', 'PCAR', 'SNDK', 'SYMC', 'WFM', 'XRAY', 'YHOO']

    source = 'yahoo'
    enddate = datetime.datetime.today()
    period = datetime.timedelta(days=np.floor(365 * 1))
    startdate = datetime.datetime.today() - period

    pca_analysis = PCA()


    port_data = pca_analysis.dl_data(tickers, source, startdate, enddate)
    ratio_df, eigenvectors = pca_analysis.cal_negeigvals_ratio(port_data)

    nasdaq = pca_analysis.dl_data('^NDX', source, startdate, enddate)

    ratio_df = pca_analysis.find_neg_ratio(port_data, ratio_df)
    # plt.plot(ratio_df.index, ratio_df['port_data'])
    # plt.plot(ratio_df.index, ratio_df['neg5'], 'o', markersize=5)
    # plt.show()
    # pca_analysis.plot_ratio()
    # pca_analysis.plot_PCA()

    fig_PCA = figure(title=" Points of Interest based on PCA ",
                     y_axis_label='Close',
                     x_axis_label='Date',
                     plot_width=700, plot_height=480,
                     x_axis_type="datetime", toolbar_location=None)


    # print ratio_df['port_data'].max()
    neg5 = ratio_df['neg5'].dropna()
    portsum = ratio_df['port_data'].dropna()
    # print (neg5.values)
    fig_PCA.circle(neg5.index, neg5.values, size = 5, color='red')
    fig_PCA.line(portsum.index, portsum.values, color='gray')

    egv = pd.DataFrame(eigenvectors, columns=port_data.columns)
    # egv.reset_index(inplace=True)
    # egvm = pd.melt(egv, var_name='tic', value_name='eig', id_vars=['index'])
    # source = ColumnDataSource(data=dict(index=egvm['index'], tic=egvm['tic'],
    #                                     eig=egvm['eig']))
    #
    p = figure(x_range=(0, 18), y_range=(0,18))
    p.image(image=[eigenvector], x=0, dw=18, y=0, dh=18, palette="Spectral10")



    output_file("brewer.html")
    show(p)

    return 0


if __name__=='__main__':
    main()