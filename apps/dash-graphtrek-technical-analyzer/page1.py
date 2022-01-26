from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from utils import Header, get_stock_price, get_title, display_chart, display_analyzer
symbols = ['^TNX','QQQ','SPY','IWM','^VIX','TSLA','VTI','XLE','XLF','TQQQ']

symbols += ['ABBV','AFRM','AMD','AMZN','APPS','ASTR','ATVI','BNGO',
            'CAT','CCL','CHWY','COST','CRM',
            'DIA','DIS','DKNG','ETSY','FFND','HOG','HUT','JETS','LOGI',
            'LVS','MSFT','MU','NCLH','NFLX','NKE','NVDA','PLTR','PYPL','XLNX',
            'RBLX','RKLB','SNAP','SOFI','SQ','TWTR','U','UBER','WFC','WBA','V']

symbols += ['AAPL','ARKG','ARKK','ARKQ','BA','CHPT','COIN','DDOG',
            'DOCU','EA','FB','GOOGL']

symbols += ['MA','MP','MRNA','MSTR','NNDM']

symbols  += ['PFE','PINS','ROKU','SBUX','SHOP','SOXL','SOXX']

symbols += ['TDOC','TEN','TGT','TLT','TTD','UAA',
            'VALE','WMT','WYNN','ZM']

symbols += ['PENN','QCOM','LCID','AAL']

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
                                            {'label': '{}'.format(i), 'value': i} for i in symbols
                                        ]

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
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page"
    )



@app.callback(
    [Output('page1-display-value', 'children'), Output('vix', 'children'), Output('qqq', 'children')],
    [Input('page1-dropdown', 'value')])
def display_value(symbol):
    if symbol is None:
        symbol = "TSLA"
    df_vix_graph = get_stock_price("^VIX", "2020-01-01")
    df_qqq_graph = get_stock_price("QQQ", "2020-01-01")
    df_xxx_graph = get_stock_price(symbol, "2020-01-01")
    qqq_div = html.Div(
        [
            html.H6(get_title("QQQ", df_qqq_graph), className="subtitle padded"),
            dcc.Graph(
                id="graph-qqq",
                figure=display_chart(df_qqq_graph),
                config={"displayModeBar": False},
            )
        ]
    )

    vix_div = html.Div([
        html.H6(get_title("VIX", df_vix_graph), className="subtitle padded"),
        dcc.Graph(
            id="graph-vix",
            figure=display_chart(df_vix_graph),
            config={"displayModeBar": False},
        )
    ])

    xxx_div = html.Div([
        html.H6(get_title(symbol, df_xxx_graph), className="subtitle padded"),
        dcc.Graph(
            id="graph-xxx",
            figure=display_analyzer(symbol,df_xxx_graph),
            config={"displayModeBar": False},
        )
    ])

    return xxx_div, vix_div, qqq_div

