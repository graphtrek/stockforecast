from dash import dcc
from dash import html
import yfinance as yf

from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.momentum import RSIIndicator
import numpy as np
import plotly.graph_objs as go

import pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta

six_months = date.today() + relativedelta(months=-6)
one_month = date.today() + relativedelta(months=+1)

start_date = six_months.strftime("%Y-%m-%d")
end_date = one_month.strftime("%Y-%m-%d")

def Header(app):
    return html.Div([get_header(app)])
    #return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            # html.Div(
            #     [
            #         html.A(
            #             html.Img(
            #                 src=app.get_asset_url("dash-financial-logo.png"),
            #                 className="logo",
            #             ),
            #             href="https://plotly.com/dash",
            #         ),
            #         html.A(
            #             html.Button(
            #                 "Enterprise Demo",
            #                 id="learn-more-button",
            #                 style={"margin-left": "-10px"},
            #             ),
            #             href="https://plotly.com/get-demo/",
            #         ),
            #         html.A(
            #             html.Button("Source Code", id="learn-more-button"),
            #             href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-financial-report",
            #         ),
            #     ],
            #     className="row",
            # ),
            html.Div(
                [
                    html.Div(
                        [html.H5("Graphtrek Technical Analyzer")],
                        className="seven columns main-title",
                    ),
                    # html.Div(
                    #     [
                    #         dcc.Link(
                    #             "Full View",
                    #             href="/dash-graphtrek-technical-analyzer/full-view",
                    #             className="full-view-link",
                    #         )
                    #     ],
                    #     className="five columns",
                    # ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/dash-graphtrek-technical-analyzer/page1",
                className="tab first",
            ),
            dcc.Link(
                "Prediction",
                href="/dash-graphtrek-technical-analyzer/page2",
                className="tab",
            )
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


def indicators(df):
    # MACD
    macd = MACD(close=df['Close'],
                window_slow=26,
                window_fast=12,
                window_sign=9)
    # stochastics
    stoch = StochasticOscillator(high=df['High'],
                                 close=df['Close'],
                                 low=df['Low'],
                                 window=14,
                                 smooth_window=3)

    rsi = RSIIndicator(close=df['Close'], window=14)
    return macd, stoch, rsi


def is_support(df,i):
    cond1 = df['Low'][i] < df['Low'][i-1]
    cond2 = df['Low'][i] < df['Low'][i+1]
    cond3 = df['Low'][i+1] < df['Low'][i+2]
    cond4 = df['Low'][i-1] < df['Low'][i-2]
    return (cond1 and cond2 and cond3 and cond4)


def is_resistance(df,i):
    cond1 = df['High'][i] > df['High'][i-1]
    cond2 = df['High'][i] > df['High'][i+1]
    cond3 = df['High'][i+1] > df['High'][i+2]
    cond4 = df['High'][i-1] > df['High'][i-2]
    return (cond1 and cond2 and cond3 and cond4)


def is_far_from_level(value, levels, df):
    ave =  np.mean(df['High'] - df['Low'])
    return np.sum([abs(value - level) < ave for level in levels]) == 0


def find_nearest_greater_than(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff<0] = np.inf
    idx = diff.argmin()
    return inputData[idx]


def find_nearest_less_than(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff>0] = -np.inf
    idx = diff.argmax()
    return inputData[idx]


def get_stock_price(ticker_name, from_date):
    ticker = yf.Ticker(ticker_name)
    df = yf.download(ticker.ticker, start=from_date, interval="1d")
    #df = df.rename(columns={"Close": "Close1", "Adj Close": "Close"})

    #ticker = yf.Ticker(symbol)

    #df = ticker.history(start=from_date, interval="1d")
    #print(df.info())
    df['Date'] = pd.to_datetime(df.index)
    #df['Date'] = df['Date'].apply(mpl_dates.date2num)
    #df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MA20'] =  df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['MA300'] = df['Close'].rolling(window=200).mean()

    macd, soch, rsi = indicators(df)
    df['RSI'] = rsi.rsi().to_numpy()
    df['MACD_DIFF'] = macd.macd_diff().to_numpy()
    df['MACD'] = macd.macd().to_numpy()
    df['MACD_SIGNAL'] = macd.macd_signal().to_numpy()
    df.to_csv('/home/nexys/graphtrek/stock/' + ticker.ticker + '.csv', index=False)
    print('Get Stock Price', ticker.ticker, 'done.')
    return df


def calculate_levels(chart_df):
    levels = []
    low = 0
    high = np.round(chart_df['Close'].max(),1)
    last_day_df = chart_df[-1:]
    last_date = last_day_df['Date'].index[0].date()
    close_price = np.round(last_day_df['Close'][0],1)
    for i in range(2,len(chart_df)-2):
        try:
            if is_support(chart_df,i):
                low = chart_df['Low'][i]
            if is_far_from_level(low, levels, chart_df):
                levels.append(low)
            elif is_resistance(chart_df,i):
                high = chart_df['High'][i]
            if is_far_from_level(high, levels, chart_df):
                levels.append(high)
        except:
            print('calculate_levels error')
    levels = sorted(levels, reverse=True)

    min_level = np.round(find_nearest_less_than(close_price, levels), 1)
    if min_level > close_price:
        min_level = np.round(close_price * 0.8,1)

    max_level = np.round(find_nearest_greater_than(close_price, levels), 1)
    if max_level < close_price:
        max_level = np.round(close_price * 1.2, 1)

    print('Calculate Levels close_price', close_price, 'min_level:', min_level, 'max_level:', max_level)
    return levels, close_price, min_level, max_level


def get_title(name, df):
    last_day_df = df.iloc[-1:]
    last_date = last_day_df['Date'].dt.strftime('%Y-%m-%d')
    close_price = np.round(float(df.iloc[-1:]['Close']), 1)

    ath = np.round(float(df['Close'].max()))
    discount = np.round(ath - close_price, 1)
    discount_percent = np.round((discount / close_price) * 100, 1)
    title = name + " " + last_date + " Last Price:" + str(close_price) + "$ " + " ATH:" + str(
        ath) + "$ Discount:" + str(discount) + "$ (" + str(discount_percent) + "%)"

    return title


def display_chart(df):
    fig = go.Figure(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        line={"color": "#97151c"},
        mode="lines"
    ))
    # zoom_df = df.iloc['Date' >= start_date]

    zoom_df = df[df.Date >= start_date]
    y_zoom_max = zoom_df["High"].max()
    y_zoom_min = zoom_df["Low"].min()

    fig.update_layout(
        autosize=True,
        #        width=700,
        height=200,
        font={"family": "Raleway", "size": 10},
        margin={
            "r": 30,
            "t": 30,
            "b": 30,
            "l": 30,
        },
        showlegend=False,
        dragmode='pan',
        titlefont={
            "family": "Raleway",
            "size": 10,
        },
        xaxis={
            #            "autorange": True,
            "range": [
                start_date,
                end_date
            ],
            "rangeselector": {
                "buttons": [
                    {
                        "count": 1,
                        "label": "1M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 3,
                        "label": "3M",
                        "step": "month",
                        "stepmode": "backward"

                    },
                    {
                        "count": 6,
                        "label": "6M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 1,
                        "label": "1Y",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "label": "All",
                        "step": "all",
                    },
                ]
            },
            "showline": True,
            "type": "date",
            "zeroline": False
        }
    )

    fig.update_layout(xaxis_rangeslider_visible=False)
    #    fig.update_xaxes(type="date", range=[start_date, end_date])
    fig.update_yaxes(range=[y_zoom_min, y_zoom_max])
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    return fig


def display_analyzer(symbol,df):
    fig = go.Figure(go.Candlestick(x=df['Date'],
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'],
                                   name=symbol,
                                   showlegend=True))
    # zoom_df = df.iloc['Date' >= start_date]

    zoom_df = df[df.Date >= start_date]
    y_zoom_max = zoom_df["High"].max()
    y_zoom_min = zoom_df["Low"].min()

    fig.update_layout(
        autosize=True,
        #        width=700,
        height=400,
        font={"family": "Raleway", "size": 10},
        margin={
            "r": 30,
            "t": 30,
            "b": 30,
            "l": 30,
        },
        showlegend=True,
        dragmode='pan',
        titlefont={
            "family": "Raleway",
            "size": 10,
        },
        xaxis={
            #            "autorange": True,
            "range": [
                start_date,
                end_date
            ],
            "rangeselector": {
                "buttons": [
                    {
                        "count": 1,
                        "label": "1M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 3,
                        "label": "3M",
                        "step": "month",
                        "stepmode": "backward"

                    },
                    {
                        "count": 6,
                        "label": "6M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 1,
                        "label": "1Y",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "label": "All",
                        "step": "all",
                    },
                ]
            },
            "showline": False,
            "type": "date",
            "zeroline": False
        }
    )

    # add moving average traces
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MA20'],
                             line=dict(color='lightgreen', width=2),
                             fill=None,
                             mode='lines',
                             name='MA 20'))
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['EMA21'],
                             fill='tonexty',
                             mode='lines',
                             line=dict(color='green', width=2),
                             name='EMA 21'))
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MA50'],
                             line=dict(color='blue', width=2),
                             name='MA 50'))
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MA100'],
                             line=dict(color='orange', width=2),
                             name='MA 100'))
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MA200'],
                             line=dict(color='red', width=2),
                             name='MA 200'))
    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MA300'],
                             line=dict(color='black', width=2),
                             name='MA 300'))

    levels, close_price, min_level, max_level = calculate_levels(df)

    ath_percent = 0

    for idx, level in enumerate(levels):
        percent = 0
        if idx == 0:
            ath = level
        current_level = level
        if idx > 0:
            prev_level = levels[idx-1]
            diff = prev_level - current_level
            ath_diff = ath - current_level
            percent = (diff / current_level) * 100
            ath_percent = (ath_diff / current_level) * 100
        if level <= (min_level * 0.98) or level >= (max_level * 1.02):
            line_color = 'rgba(100, 10, 100, 0.2)'
            line_fill = None
        else:
            line_color = 'rgba(128,128,128,1)'
            line_fill = 'tonexty'
        fig.add_trace(go.Scatter(
            x = [df['Date'].min(), df['Date'].max()],
            y = [level, level],
            mode="lines+text",
            name="Lines and Text",
            fill=line_fill,
            showlegend=False,
            text=['', '$' + str(np.round(current_level, 1)) + ' (' + str(np.round(percent, 1)) + '% disc:' + str(np.round(ath_percent, 1)) + '%)', ''],
            textposition="top right",
            line=dict(shape='linear', color=line_color, dash='dash', width=1)
        ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    #    fig.update_xaxes(type="date", range=[start_date, end_date])
    fig.update_yaxes(range=[y_zoom_min, y_zoom_max])
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    return fig