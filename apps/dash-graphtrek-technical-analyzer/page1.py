from dash import dcc
from dash import html
import yfinance as yf
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from app import app
from symbols import symbols
from utils import (
    Header,
    get_stock_price,
    get_stock_price1,
    get_predict_price,
    get_title,
    display_chart,
    display_analyzer,
    get_symbols_info_df,
    get_text,
    make_dash_table,
    make_options_table,
    find_level_option_interests,
    calculate_levels,
    get_last_price,
    get_predictions,
    load_options_df
)


symbols_info_df = get_symbols_info_df(symbols)


def get_options_label(row):
    return row['Symbol'] \
            + " " + get_text("", row['Name'], "") \
            + " " + get_text("ShortRatio:", row['ShortRatio'], "") \
            + " " + get_text("", row['Recommendation'], "").upper() \
            + " " + get_text("TargetPrice:", row['TargetPrice'], "") \
            + " " + get_text("Earning:", row['Earning'], "") \
            + " " + get_text("in ", row['Day'], " days") \
            + " " + get_text("Sector:", row['Sector'], "") \
            + " " + get_text("Industry:", row['Industry'], "")


layout = html.Div(
    [
        Header(app),
        # page1
        html.Div(
            [
                # Row
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id='page1-dropdown',
                                    options=[
                                        {'label': get_options_label(row),
                                         'value': row['Symbol']} for index, row in symbols_info_df.iterrows()
                                    ],
                                    persistence=True, value="TSLA"
                                )
                            ]
                        )
                    ],
                    className="row"
                ),
                # Row
                html.Div(
                    [
                        html.Div(id="vix", className="six columns"),
                        html.Div(id="spy", className="six columns"),
                    ],
                    className="row ",
                ),
                # Row
                html.Div(
                    [
                        html.Div(id='page1-main-chart'),
                        html.Div(id="page1-wheel-table")
                    ],
                    className="row ",
                ),
                # Row
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="calls_title"),
                                html.Div(id="calls")
                            ], className="six columns"
                        ),
                        html.Div(
                            [
                                html.Div(id="puts_title"),
                                html.Div(id="puts")
                            ], className="six columns"
                        ),
                    ],
                    className="row ",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Header"),
                        dbc.ModalBody("This is the content of the modal"),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close", className="ml-auto")
                        ),
                    ],
                    id="modal",
                )
            ],
            className="sub_page",
        ),
    ],
    className="page"
)

vix_ticker = yf.Ticker("^VIX")
spy_ticker = yf.Ticker("SPY")


@app.callback(
    [Output('symbol', 'children'),
     Output('page1-main-chart', 'children'),
     Output('page1-wheel-table', 'children'),
     Output('vix', 'children'),
     Output('spy', 'children'),
     Output('calls_title', 'children'),
     Output('calls', 'children'),
     Output('puts_title', 'children'),
     Output('puts', 'children')],
    [Input('page1-dropdown', 'value')])
def display_value(symbol):
    if symbol is None:
        symbol = "TSLA"
    ticker = yf.Ticker(symbol)
    df_vix_graph = get_stock_price1(vix_ticker, "2021-01-01")
    df_spy_graph = get_stock_price1(spy_ticker, "2021-01-01")
    df_xxx_graph = get_stock_price(ticker, "2020-01-01")

    calls_class_name = "subtitle"
    puts_class_name = "subtitle"
    xxx_class_name = "subtitle"

    close_price, last_date, prev_close_price = get_last_price(df_xxx_graph)
    change = np.round(close_price - prev_close_price, 2)
    change_percent = np.round((change / prev_close_price) * 100, 1)
    if change > 0:
        symbol_view = html.B(
            symbol + " $" + str(close_price) + " (" + str(change_percent) + "%" + ") " + str(change),
            className="symbol_view_green",
            )
    else:
        symbol_view = html.B(
            symbol + " $" + str(close_price) + " (" + str(change_percent) + "%" + ") " + str(change),
            className="symbol_view_red",
            )

    levels, close_price, min_level, max_level = calculate_levels(df_xxx_graph)

    options_df = load_options_df(symbol)
    put_options_df = pd.DataFrame()
    call_options_df = pd.DataFrame()

    wheel_put_options_df, wheel_call_options_df = \
        find_level_option_interests(options_df, min_level, max_level, 0, 21)

    near_put_options_df, near_call_options_df = \
        find_level_option_interests(options_df, min_level, max_level, 21, 45)

    mid_put_options_df, mid_call_options_df = \
        find_level_option_interests(options_df, min_level, max_level, 45, 76)

    # far_put_options_df, far_call_options_df = \
    #     find_level_option_interests(options_df, min_level, max_level, 76, 120)

    sum_call_options = 0
    call_options_df = call_options_df.append(wheel_call_options_df)
    call_options_df = call_options_df.append(near_call_options_df)
    call_options_df = call_options_df.append(mid_call_options_df)
    # call_options_df = call_options_df.append(far_call_options_df)
    if len(call_options_df) > 0:
        call_options_df = call_options_df.sort_values(by=['dte'])
        sum_call_options = int(sum(call_options_df[['openInterest', 'volume']].sum(axis=1)))

    calls_bull = False
    puts_bull = False
    sum_put_options = 0
    put_options_df = put_options_df.append(wheel_put_options_df)
    put_options_df = put_options_df.append(near_put_options_df)
    put_options_df = put_options_df.append(mid_put_options_df)
    # put_options_df = put_options_df.append(far_put_options_df)
    if len(put_options_df) > 0:
        put_options_df = put_options_df.sort_values(by=['dte'])
        sum_put_options = int(sum(put_options_df[['openInterest', 'volume']].sum(axis=1)))

    all_options = sum_call_options + sum_put_options
    if all_options > 0:
        put_options_percent = np.round((sum_put_options / all_options) * 100, 1)
        call_options_percent = np.round((sum_call_options / all_options) * 100, 1)
        # max_calls_open_interest_index = call_options_df["openInterest"].idxmax()
        # calls_strike = call_options_df.loc[max_calls_open_interest_index]["strike"]
        calls_strike = np.mean(call_options_df["strike"])

        # max_puts_open_interest_index = put_options_df["openInterest"].idxmax()
        # puts_strike = put_options_df.loc[max_puts_open_interest_index]["strike"]
        puts_strike = np.mean(put_options_df["strike"])

        if call_options_percent > 60 or calls_strike >= close_price:
            calls_class_name = "subtitle_green"
            calls_bull = True
        if puts_strike >= close_price:
            puts_class_name = "subtitle_green"
            puts_bull = True

        calls_title = html.H6(
            ["CALL" + " " + str(call_options_percent) + "%" + " (" + '{:,}'.format(sum_call_options) + ")"],
            className=calls_class_name + " padded")
        puts_title = html.H6(
            ["PUT" + " " + str(put_options_percent) + "%" + " (" + '{:,}'.format(sum_put_options) + ")"],
            className=puts_class_name + " padded")
    else:
        calls_title = html.H6()
        puts_title = html.H6(),

    spy_div = html.Div(
        [
            # html.H6(get_title("SPY", df_spy_graph), className="subtitle padded"),
            dcc.Graph(
                id="graph-spy",
                figure=display_chart(spy_ticker, df_spy_graph),
                config={"displayModeBar": False, "scrollZoom": True},
            )
        ]
    )

    vix_div = html.Div([
        # html.H6(get_title("VIX", df_vix_graph), className="subtitle padded"),
        dcc.Graph(
            id="graph-vix",
            figure=display_chart(vix_ticker, df_vix_graph),
            config={"displayModeBar": False, "scrollZoom": True},
        )
    ])

    indicators_test_prediction_df, indicators_prediction_df = get_predictions(symbol)

    predict_price = close_price
    predictions_bull = False
    if indicators_prediction_df is not None:
        predict_price, first_prediction, mean_prediction = \
            get_predict_price(df_xxx_graph, indicators_prediction_df, max_level, min_level)
        if mean_prediction >= first_prediction:
            predictions_bull = True

    if puts_bull and calls_bull and predictions_bull:
        xxx_class_name = "subtitle_green"

    xxx_div = html.Div(
        [
            html.H6([get_title(ticker, df_xxx_graph, predict_price),
                     " ",
                     html.A("TradingView",
                            href='https://in.tradingview.com/chart?symbol=' + symbol,
                            style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold'},
                            target="_blank"),
                     " ",
                     html.A("SeekingAlpha",
                            href='https://seekingalpha.com/symbol/' + symbol,
                            style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold'},
                            target="_blank")

                     ],
                    className=xxx_class_name + " padded"),
            dcc.Graph(
                id="graph-xxx",
                figure=display_analyzer(symbol, df_xxx_graph, indicators_test_prediction_df, indicators_prediction_df),
                config={"displayModeBar": False, "scrollZoom": True},
            )
        ],
        className="ten columns")

    tables = []
    for index, row in put_options_df.iterrows():
        expiration = row['expirationDate']
        strike = row['strike']
        mid = (float(row['ask']) + float(row['bid'])) / 2
        dte = row["dte"]
        premium = np.round(mid * 100, 2)
        sell_price = np.round(float(strike) - mid, 2)
        buy_price = np.round(float(strike) + mid, 2)

        discount_percent = np.round(100 - ((close_price / sell_price) * 100),2)
        wheel_df = pd.DataFrame({
            "label": ["Stock Price", "Expiration", "Strike", "Premium", "Sell", "Buy"],
            "value": [
                "$" + str('{:,}'.format(close_price)),
                expiration + " in " + str(dte) + " days",
                "$" + str('{:,}'.format(strike)),
                "$" + str('{:,}'.format(premium)),
                "$" + str('{:,}'.format(sell_price)) + " (" + str(discount_percent) + "%)",
                "$" + str('{:,}'.format(buy_price))
            ]
        })
        tables.append(html.Table(make_dash_table(wheel_df)))

    wheel_div = html.Div(
        [
            html.H6(
                ["Puts"], className=puts_class_name + " padded"
            ),
            html.Div(children=tables)
        ],
        className="two columns",
    )

    calls_table = html.Table(make_options_table(call_options_df))
    puts_table = html.Table(make_options_table(put_options_df))
    return symbol_view, xxx_div, wheel_div, vix_div, spy_div, calls_title, calls_table, puts_title, puts_table


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
