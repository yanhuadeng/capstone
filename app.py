from scripts import FinanceWhatIf
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
# Bootstrap(app)

@app.route('/')
def main():
    return redirect('/index3')

# Main Process
@app.route('/index3')
def stock():
    stockcode = request.args.get('stockcode', 'GOOG,AAPL', type=str)
    indexticker = request.args.get('indexticker', '^GSPC', type=str)
    deltaInd = request.args.get('delta', 0, type=int)
    stockperiod = request.args.get('period', 1.0, type=float)

    stockcode = stockcode.strip('').split(",")
    stockcode.insert(0, indexticker)
    source = "yahoo"
    deltaInd = deltaInd
    print stockcode
    portRetPast, portRetFuture = FinanceWhatIf.FinanceWhatIf(stockcode, source, stockperiod, deltaInd).portfolioPlot()


    fig = figure(title="Total Portfolio Returns",
                 y_axis_label='Returns',
                 x_axis_label = 'Date',
                 plot_width=600, plot_height=480,
                 x_axis_type="datetime", toolbar_location=None)
    fig.line(portRetPast.index, portRetPast["Returns"], color="gray")
    fig.line(portRetFuture.index, portRetFuture["Returns"], color="red")

    jsresources = INLINE.render_js()
    cssresources = INLINE.render_css()
    script, div = components(fig, INLINE)

    html = render_template('index3.html',
                           plot_script=script, plot_div=div,
                           js_resources=jsresources, css_resources=cssresources)

    return html

if __name__ == '__main__':
    app.run( port=33500,debug=True)
    #testPort()