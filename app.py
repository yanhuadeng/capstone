from scripts import FinanceWhatIf
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from flask import Flask, render_template, request, redirect, send_file
from io import BytesIO
import pandas as pd
from itertools import ifilter
import matplotlib.pyplot as plt
import base64


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

# Main Process
@app.route('/index3')
def stock():
    stockcodeStr = request.args.get('stockcode', 'GOOG, FB', type=str)
    indexticker = request.args.get('indexticker', '^GSPC', type=str)
    deltaInd = request.args.get('delta', 0.01, type=int)
    stockperiod = request.args.get('period', 1.0, type=float)

    stockcode = stockcodeStr.replace(' ', '').strip('').split(",")
    stockcode.insert(0, indexticker)
    source = "yahoo"
    deltaInd = deltaInd
    portRet = FinanceWhatIf.FinanceWhatIf(stockcode, source, stockperiod, deltaInd)
    portRetPast, portRetFuture = portRet.portfolioPlot()
    fig = figure(title="Total Portfolio Returns for "+ stockcodeStr,
                 y_axis_label='Returns',
                 x_axis_label = 'Date',
                 plot_width=700, plot_height=480,
                 x_axis_type="datetime", toolbar_location=None)
    fig.line(portRetPast.index, portRetPast["Returns"], color="gray")
    fig.line(portRetFuture.index, portRetFuture["Returns"], color="red")
    fig.line(portRetFuture.index[[0,-1]],[portRet.startPtVal, portRet.endPtVal], line_dash=[4, 4], color='red')

    jsresources = INLINE.render_js()
    cssresources = INLINE.render_css()
    script, div = components(fig, INLINE)
    coefficients = ((portRet.regCoeff).ix[1:].T.to_dict()).values()

    def regressionPlots(portRet):
        x = portRet.retDF[portRet.indexticker]
        Nplt = len(portRet.ticker) - 1
        for n in range(Nplt):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x, portRet.retDF[portRet.retDF.columns[n]], 'o', label="data")
            ax.plot(x, portRet.olsDict[portRet.retDF.columns[n]].fittedvalues, 'r--.')
            plt.ylabel(portRet.ticker[n+1])
            plt.xlabel(portRet.indexticker)
            # fig.savefig('pltfigs/fig'+str(n)+'.jpeg')

    regressionPlots(portRet)

    html = render_template('index3.html', coefficients=coefficients,
                           plot_script=script, plot_div=div,
                           js_resources=jsresources, css_resources=cssresources)

    return html



if __name__ == '__main__':
    app.run(port=33500, debug=True)
    #testPort()