from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from flask import Flask, render_template, request, redirect
from bokeh.palettes import brewer

import numpy as np

from scripts import PortfolioAnalysis, PCA



app = Flask(__name__)
# Bootstrap(app)

@app.route('/')
def main():
    return redirect('/index')


# Main Process
@app.route('/index')
def stock():
    stockcodeStr = request.args.get('stockcode', 'GOOG, FB, AAPL', type=str)
    indexticker = [request.args.get('indexticker', '^GSPC', type=str)]

    deltaInd = request.args.get('delta', 0.01, type=int)
    stockperiod = request.args.get('period', 1, type=float)

    stockcode = stockcodeStr.replace(' ', '').strip().upper().split(",")
    source = "yahoo"
    deltaInd = deltaInd

    PA = PortfolioAnalysis.PortfolioAnalysis(stockcode, source, stockperiod, indexticker)


    PA.dl_data(PA.tickersidx, PA.source, PA.startdate, PA.enddate)

    PA.compute_regression()

    # Portfolio Correlations
    fig_corr = figure(title="Linear Regression Analysis for " + stockcodeStr,
                      x_axis_label='Factor',
                      y_axis_label='Components',
                      plot_width=700, plot_height=480,
                      toolbar_location=None)
    colors = brewer["Spectral"][11]
    for n in range(len(stockcode)):
        fig_corr.scatter(PA.idx_daily_ret.values.T[0], PA.port_daily_ret.ix[:, n].values,
                         radius=.001 * PA.port_daily_ret.max().max(), fill_color='b', fill_alpha=0.5,
                         legend=stockcode[n])
        fig_corr.line(PA.idx_daily_ret.values.T[0], PA.ols_dict.fittedvalues.ix[:, n].values, color='b')
    script_corr, div_corr = components(fig_corr, INLINE)


    # Portfolio risk plots
    # risks, returns, allocs = PA.port_opt_classic(PA.port_daily_ret)
    risksrand, returnsrand, weights= PA.port_alloc_rand(PA.port_daily_ret)

    maxindex = np.argmax(returnsrand)
    retmax = returnsrand[maxindex]
    riskmax = risksrand[maxindex]
    allocs = weights[maxindex]

    fig_risk = figure(title="Risks and Returns for " + stockcodeStr,
                     y_axis_label='Returns',
                     x_axis_label='Risks',
                     plot_width=700, plot_height=480,
                     toolbar_location=None)
    # fig_risk.line(risks, returns, color='red')
    fig_risk.scatter(risksrand, returnsrand,  fill_color = 'red', fill_alpha=0.5) #radius=0.01*max(returnsrand),
    script_risk, div_risk = components(fig_risk, INLINE)

    # PCA
    PCA_analysis = PCA.PCA()
    ratio_df, eigenvectors = PCA_analysis.cal_negeigvals_ratio(PA.port_data)
    ratio_df = PCA_analysis.find_neg_ratio(PA.port_data, ratio_df)

    fig_PCA = figure(title=" Points of Interest based on PCA ",
                     y_axis_label='Close',
                     x_axis_label='Date',
                     plot_width=700, plot_height=480,
                     x_axis_type="datetime", toolbar_location=None)

    # print ratio_df['port_data'].max()
    neg5 = ratio_df['neg5'].dropna()
    portsum = ratio_df['port_data'].dropna()
    fig_PCA.circle(neg5.index, neg5.values, size=5, color='red')
    fig_PCA.line(portsum.index, portsum.values, color='gray')

    script_PCA, div_PCA = components(fig_PCA, INLINE)

    fig_eig = figure(title=" Slice of Eigenvectors of the Portfolio ",
                     y_axis_label='Num of Components',
                     x_axis_label='Components',
                     x_range=(0, len(eigenvectors)),
                     y_range=(0, len(eigenvectors)),
                     plot_width=500, plot_height=500,
                     toolbar_location=None)

    fig_eig.image(image=[eigenvectors], x=0, dw=len(eigenvectors),
            y=0, dh=len(eigenvectors), palette="Spectral10")
    script_eig, div_eig = components(fig_eig, INLINE)


    fig_neg = figure(title="Ratio of Negative Values in 1-PCA Eigenvectors ",
                     y_axis_label='Ratio %',
                     x_axis_label = 'Date',
                     plot_width=700, plot_height=480,
                     y_range = (0, 100),
                     x_axis_type="datetime", toolbar_location=None)
    fig_neg.line(ratio_df.index, ratio_df["ratio"]*100, color="blue")
    fig_neg.line(ratio_df.index, ratio_df["ratio_n"]*100, color="gray")
    script_neg, div_neg = components(fig_neg, INLINE)

    #Port prediction based on maximum risk tolerance
    past, future = PA.cal_prediction(deltaInd = 0.01)
    fig = figure(title="Total Portfolio Returns for "+ stockcodeStr,
                 y_axis_label='Close',
                 x_axis_label = 'Date',
                 plot_width=700, plot_height=500,
                 x_axis_type="datetime", toolbar_location=None)
    fig.line(past.index, past["Returns"], color="gray")
    fig.line(future.index, future["Returns"], color="red")
    fig.line(future.index[[0,-1]],[PA.startPtVal, PA.endPtVal], line_dash=[4, 4], color='red')
    script, div = components(fig, INLINE)

    jsresources = INLINE.render_js()
    cssresources = INLINE.render_css()
    coefficients = ((PA.reg_coefficients).ix[1:].T.to_dict()).values()

    # allocations = (allocs[0].reshape(1, len(allocs[0])))[0]
    # allocations_dict = dict(zip(stockcode, allocations*100))
    allocations_dict = dict(zip(stockcode, allocs * 100))

    html = render_template('index.html', coefficients=coefficients,
                           # port_opt_params = (risks[0]*100, returns[0]*100, allocations_dict),
                           port_opt_params = (riskmax*100, retmax*100, allocations_dict),
                           corr_script=script_corr, corr_div=div_corr,
                           plot_script=script, plot_div=div,
                           risk_script=script_risk, risk_div=div_risk,
                           PCA_script=script_PCA, PCA_div=div_PCA,
                           eig_script=script_eig, eig_div=div_eig,
                           neg_script=script_neg, neg_div=div_neg,
                           js_resources=jsresources, css_resources=cssresources,
                           status={'code': 0, 'msg': ''})
    return html

if __name__ == '__main__':
    app.run(port=33350, debug=True)
    #testPort()