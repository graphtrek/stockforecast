from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from utils import Header, get_stock_price, get_title, display_chart

layout = html.Div(
        [
            Header(app),
            # page1
            html.Div(
                [
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
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id='page1-dropdown',
                                            options=[
                                                {'label': 'Page 1 - {}'.format(i), 'value': i} for i in [
                                                    'NYC', 'MTL', 'LA'
                                                ]
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
                            html.Div(id='page1-display-value', className="six columns")
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
def display_value(value):
    df_vix_graph = get_stock_price("^VIX", "2018-01-01")
    df_qqq_graph = get_stock_price("QQQ", "2018-01-01")

    qqq_div = html.Div(
        [
            html.H6(get_title("QQQ", df_qqq_graph), className="subtitle padded"),
            dcc.Graph(
                id="graph-2",
                figure=display_chart(df_qqq_graph),
                config={"displayModeBar": False},
            )
        ]
    )

    vix_div = html.Div([
        html.H6(get_title("VIX", df_vix_graph), className="subtitle padded"),
        dcc.Graph(
            id="graph-x",
            figure=display_chart(df_vix_graph),
            config={"displayModeBar": False},
        )
    ])
    return vix_div, vix_div, qqq_div

