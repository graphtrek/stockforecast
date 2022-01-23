import dash_html_components as html
import dash_core_components as dcc
import yfinance as yf

from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.momentum import RSIIndicator

import pandas as pd

def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


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
                        [html.H5("Graphtrek Technical Analyser")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/dash-financial-report/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
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
                href="/dash-financial-report/marketOverview",
                className="tab first",
            ),
            dcc.Link(
                "Prediction",
                href="/dash-financial-report/prediction-charts",
                className="tab",
            ),
            dcc.Link(
                "Portfolio & Management",
                href="/dash-financial-report/portfolio-management",
                className="tab",
            ),
            dcc.Link(
                "Fees & Minimums", href="/dash-financial-report/fees", className="tab"
            ),
            dcc.Link(
                "Distributions",
                href="/dash-financial-report/distributions",
                className="tab",
            ),
            dcc.Link(
                "News & Reviews",
                href="/dash-financial-report/news-and-reviews",
                className="tab",
            ),
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

def get_stock_price(ticker, from_date):
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