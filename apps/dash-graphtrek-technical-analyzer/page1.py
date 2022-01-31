from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from app import app
from utils import (
    Header,
    get_stock_price,
    get_title,
    display_chart,
    display_analyzer,
    get_earnings,
    get_text,
    make_dash_table,
    find_level_option_interests,
    calculate_levels
)

symbols = ['^TNX', 'QQQ', 'SPY', 'IWM', '^VIX', 'TSLA', 'VTI', 'XLE', 'XLF', 'TQQQ']
symbols += ['BTC-USD', 'ETH-USD']
symbols += ['ABBV', 'AFRM', 'AMD', 'AMZN', 'APPS', 'ASTR', 'ATVI', 'BNGO',
            'CAT', 'CCL', 'CHWY', 'COST', 'CRM',
            'DIA', 'DIS', 'DKNG', 'ETSY', 'FFND', 'HOG', 'HUT', 'JETS', 'LOGI',
            'LVS', 'MSFT', 'MU', 'NCLH', 'NFLX', 'NKE', 'NVDA', 'PLTR', 'PYPL', 'XLNX',
            'RBLX', 'RKLB', 'SNAP', 'SOFI', 'SQ', 'TWTR', 'U', 'UBER', 'WFC', 'WBA', 'V']

symbols += ['AAPL', 'ARKG', 'ARKK', 'ARKQ', 'BA', 'CHPT', 'COIN', 'DDOG',
            'DOCU', 'EA', 'FB', 'GOOGL']

symbols += ['MA', 'MP', 'MRNA', 'MSTR', 'MCD', 'NNDM']

symbols += ['PFE', 'PINS', 'ROKU', 'SBUX', 'SHOP', 'SOXL', 'SOXX']

symbols += ['TDOC', 'TEN', 'TGT', 'TLT', 'TTD', 'UAA',
            'VALE', 'WMT', 'WYNN', 'ZM']

symbols += ['PENN', 'QCOM', 'LCID', 'AAL']
symbols.sort()


def load_dropdown():
    return get_earnings(symbols)


def get_options_label(row):
    return row['Symbol'] + " " + get_text("Earning:", row['Earning'], "") + get_text(" in ", row['Day'], " days")


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
                                         'value': row['Symbol']} for index, row in load_dropdown().iterrows()
                                    ],
                                    value="TSLA"
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
                        html.Div(id="qqq", className="six columns"),
                    ],
                    className="row ",
                ),
                # Row
                html.Div(
                    [
                        html.Div(id='page1-display-value')
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


@app.callback(
    [Output('page1-display-value', 'children'),
     Output('vix', 'children'),
     Output('qqq', 'children'),
     Output('calls_title', 'children'),
     Output('calls', 'children'),
     Output('puts_title', 'children'),
     Output('puts', 'children')],
    [Input('page1-dropdown', 'value')])
def display_value(symbol):
    if symbol is None:
        symbol = "TSLA"
    df_vix_graph = get_stock_price("^VIX", "2021-01-01")
    df_qqq_graph = get_stock_price("QQQ", "2021-01-01")
    df_xxx_graph = get_stock_price(symbol, "2021-01-01")

    qqq_div = html.Div(
        [
#            html.H6(get_title("QQQ", df_qqq_graph), className="subtitle padded"),
            dcc.Graph(
                id="graph-qqq",
                figure=display_chart("QQQ",df_qqq_graph),
                config={"displayModeBar": False},
            )
        ]
    )

    vix_div = html.Div([
#        html.H6(get_title("VIX", df_vix_graph), className="subtitle padded"),
        dcc.Graph(
            id="graph-vix",
            figure=display_chart("VIX",df_vix_graph),
            config={"displayModeBar": False},
        )
    ])

    xxx_div = html.Div([
        html.H6([get_title(symbol, df_xxx_graph),
                 " ",
                 html.A("TradingView",
                        href='https://in.tradingview.com/chart?symbol=TSLA' + symbol,
                        style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold'},
                        target="_blank"),
                 " ",
                 html.A("SeekingAlpha",
                        href='https://seekingalpha.com/symbol/' + symbol,
                        style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold'},
                        target="_blank")

                 ],
                className="subtitle padded"),
        dcc.Graph(
            id="graph-xxx",
            figure=display_analyzer(symbol, df_xxx_graph),
            config={"displayModeBar": False},
        )
    ])
    levels, close_price, min_level, max_level = calculate_levels(df_xxx_graph)

    put_options_df = pd.DataFrame()
    call_options_df = pd.DataFrame()
    near_put_options_df, near_call_options_df = find_level_option_interests(symbol, min_level, max_level, 0, 45)
    far_put_options_df, far_call_options_df = find_level_option_interests(symbol, min_level, max_level, 45, 365)

    all_call_options = 0
    call_options_df = call_options_df.append(near_call_options_df)
    call_options_df = call_options_df.append(far_call_options_df)
    if len(call_options_df) > 0:
        call_options_df = call_options_df.sort_values(by=['dte'])
        all_call_options = int(sum(call_options_df[['openInterest', 'volume']].sum(axis=1)))

    all_put_options = 0
    put_options_df = put_options_df.append(near_put_options_df)
    put_options_df = put_options_df.append(far_put_options_df)
    if len(put_options_df) > 0:
        put_options_df = put_options_df.sort_values(by=['dte'])
        all_put_options = int(sum(put_options_df[['openInterest', 'volume']].sum(axis=1)))

    all_options = all_call_options + all_put_options
    if all_options > 0:
        put_options_percent = np.round((all_put_options / all_options) * 100, 1)
        call_options_percent = np.round((all_call_options / all_options) * 100, 1)
        calls_title = html.H6(
            ["CALL" + " " + str(call_options_percent) + "%" + " (" + '{:,}'.format(all_call_options) + ")"],
            className="subtitle padded")
        puts_title = html.H6(
            ["PUT" + " " + str(put_options_percent) + "%" + " (" + '{:,}'.format(all_put_options) + ")"],
            className="subtitle padded")
    else:
        calls_title = html.H6(["CALL"], className="subtitle padded")
        puts_title = html.H6(["PUT"], className="subtitle padded"),

    calls_table = html.Table(make_dash_table(call_options_df))
    puts_table = html.Table(make_dash_table(put_options_df))
    return xxx_div, vix_div, qqq_div, calls_title, calls_table, puts_title, puts_table

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
