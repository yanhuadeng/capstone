from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from flask import Flask, render_template, request, redirect, send_file
import numpy as np
from itertools import ifilter
# import matplotlib.pyplot as plt
import base64
from scripts import PortfolioAnalysis


app = Flask(__name__)
# Bootstrap(app)

@app.route('/')
def main():
    return redirect('/index3')

@app.route('/correlations')
def correlations():
    return render_template('correlations.html')


@app.route('/pltmag')
def pltmag():
    return render_template('pltmag.html')


# def regressionPlots(portRet):
#     x = portRet.retDF[portRet.indexticker]
#     Nplt = len(portRet.ticker) - 1
#     for n in range(Nplt):
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         ax.plot(x, portRet.retDF[portRet.retDF.columns[n]], 'o', label="data")
#         ax.plot(x, portRet.olsDict[portRet.retDF.columns[n]].fittedvalues, 'r--.')
#         plt.ylabel(portRet.ticker[n + 1])
#         plt.xlabel(portRet.indexticker)
        # fig.savefig('pltfigs/fig'+str(n)+'.jpeg')

# Main Process
@app.route('/index3')
def stock():
    stockcodeStr = request.args.get('stockcode', 'GOOG, FB, AAPL', type=str)
    indexticker = request.args.get('indexticker', '^GSPC', type=str)
    deltaInd = request.args.get('delta', 0.01, type=int)
    stockperiod = request.args.get('period', 0.5, type=float)

    stockcode = stockcodeStr.replace(' ', '').strip('').split(",")
    stockcode.insert(0, indexticker)
    source = "yahoo"
    deltaInd = deltaInd
    PA = PortfolioAnalysis.PortfolioAnalysis(stockcode, source, stockperiod, deltaInd)

    # Portfolio Correlations
    fig_corr = figure(title="Linear Regression Analysis for " + stockcodeStr,
                      y_axis_label='Factor',
                      x_axis_label='Components',
                      plot_width=700, plot_height=480,
                      toolbar_location=None)

    x = PA.daily_ret[PA.indexticker]
    Nplt = len(PA.tickers) - 1
    colors = ['blue', 'red', 'green', 'yellow']

    for n in range(Nplt):
        # plt.subplot(Nplt,1,n+1)
        fig_corr.scatter(x, PA.daily_ret[PA.daily_ret.columns[n]],
                         radius=0.0008, fill_color = colors[n], fill_alpha=0.2)
        fig_corr.line(x, PA.ols_dict[PA.daily_ret.columns[n]].fittedvalues, color = colors[n])
    script_corr, div_corr = components(fig_corr, INLINE)
    # Portfolio risk plots
    # risks, returns, allocs = PA.port_opt_classic(PA.daily_ret.ix[:, 0:len(stockcode) - 1])
    risksrand, returnsrand = PA.port_alloc_rand(PA.daily_ret.ix[:, 0:len(stockcode) - 1])

    fig_risk = figure(title="Risks and Returns for " + stockcodeStr,
                 y_axis_label='Returns',
                 x_axis_label='Risks',
                 plot_width=700, plot_height=480,
                 toolbar_location=None)
    # fig_risk.line(risks, returns, color='red')

    fig_risk.scatter(risksrand, returnsrand, radius=0.0001, fill_color = 'red', fill_alpha=0.2)
    script_risk, div_risk = components(fig_risk, INLINE)


    past, future = PA.cal_prediction(np.ones([1, len(stockcode)-1])/float(len(stockcode)-1)) #allocs[0].reshape(1,len(allocs[0]))
    fig = figure(title="Total Portfolio Returns for "+ stockcodeStr,
                 y_axis_label='Returns',
                 x_axis_label = 'Date',
                 plot_width=700, plot_height=480,
                 x_axis_type="datetime", toolbar_location=None)
    fig.line(past.index, past["Returns"], color="gray")
    fig.line(future.index, future["Returns"], color="red")
    fig.line(future.index[[0,-1]],[PA.startPtVal, PA.endPtVal], line_dash=[4, 4], color='red')
    script, div = components(fig, INLINE)


    jsresources = INLINE.render_js()
    cssresources = INLINE.render_css()

    coefficients = ((PA.reg_coefficients).ix[1:].T.to_dict()).values()

    # allocations = allocs[0].reshape(1, len(allocs[0]))

    html = render_template('index3.html', coefficients=coefficients,
                           corr_script=script_corr, corr_div=div_corr,
                           plot_script=script, plot_div=div,
                           risk_script=script_risk, risk_div=div_risk,
                           js_resources=jsresources, css_resources=cssresources)

    return html



if __name__ == '__main__':
    app.run(port=33500, debug=True)
    #testPort()