import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from utils import Header, make_dash_table, get_stock_price
import pandas as pd
import pathlib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


#df_current_prices = pd.read_csv(DATA_PATH.joinpath("df_current_prices.csv"))
#df_hist_prices = pd.read_csv(DATA_PATH.joinpath("df_hist_prices.csv"))
#df_avg_returns = pd.read_csv(DATA_PATH.joinpath("df_avg_returns.csv"))
#df_after_tax = pd.read_csv(DATA_PATH.joinpath("df_after_tax.csv"))
#df_recent_returns = pd.read_csv(DATA_PATH.joinpath("df_recent_returns.csv"))

df_vix_graph = get_stock_price(yf.Ticker('^VIX'),"2018-01-01")
df_tnx_graph = get_stock_price(yf.Ticker('^TNX'),"2018-01-01")
df_qqq_graph = get_stock_price(yf.Ticker('QQQ'),"2018-01-01")
df_spy_graph = get_stock_price(yf.Ticker('SPY'),"2018-01-01")
df_tsla_graph = get_stock_price(yf.Ticker('TSLA'),"2018-01-01")


def get_title(name,df):
    last_day_df = df.iloc[-1:]
    last_date = last_day_df['Date'].dt.strftime('%Y-%m-%d')
    close_price = np.round(float(df.iloc[-1:]['Close']),1)

    ath = np.round(float(df['Close'].max()))
    discount = np.round(ath - close_price,1)
    discount_percent = np.round((discount / close_price) * 100, 1)
    title = name + " " + last_date + " Last Price:" + str(close_price) + "$ " + " ATH:" + str(ath) + "$ Discount:" + str(discount) + "$ (" + str(discount_percent) + "%)"

    return title

def display_chart(df):

    fig = go.Figure( go.Scatter(
        x=df["Date"],
        y=df["Close"],
        line={"color": "#97151c"},
        mode="lines"
    ))
    start_date = "2021-06-01"
    end_date = "2022-01-31"
    #zoom_df = chart_df.iloc['Date' >= start_date]

    zoom_df = df[df.Date >= start_date]
    y_zoom_max = zoom_df["High"].max()
    y_zoom_min = zoom_df["Low"].min()

    fig.update_layout(
        autosize=True,
        width=700,
        height=200,
        font={"family": "Raleway", "size": 10},
        margin={
            "r": 30,
            "t": 30,
            "b": 30,
            "l": 30,
        },
        showlegend=False,
        dragmode= 'pan',
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
    fig.update_yaxes(range=[y_zoom_min,y_zoom_max])
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',spikedash='dash')
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    return fig



def create_layout(app):
    #df_vix_graph = get_stock_price(yf.Ticker('^VIX'),"2018-01-01")
    #df_tnx_graph = get_stock_price(yf.Ticker('^TNX'),"2018-01-01")
    #df_qqq_graph = get_stock_price(yf.Ticker('QQQ'),"2018-01-01")
    #df_spy_graph = get_stock_price(yf.Ticker('SPY'),"2018-01-01")
    #df_tsla_graph = get_stock_price(yf.Ticker('TSLA'),"2018-01-01")

    return html.Div(
        [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(get_title("VIX",df_vix_graph), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-1",
                                        figure=display_chart(df_vix_graph),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(get_title("TNX",df_tnx_graph), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-2",
                                        figure=display_chart(df_tnx_graph),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(get_title("QQQ",df_qqq_graph), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-3",
                                        figure=display_chart(df_qqq_graph),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(get_title("SPY",df_spy_graph), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4",
                                        figure=display_chart(df_spy_graph),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(get_title("TSLA",df_tsla_graph), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-5",
                                        figure=display_chart(df_tsla_graph),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
