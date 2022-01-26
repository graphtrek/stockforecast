from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
import page1
from utils import get_stock_price


def serve_layout():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])


app.layout = serve_layout


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dash-graphtrek-technical-analyzer/page1':
        page1.df_qqq_graph = get_stock_price("QQQ","2018-01-01")
        page1.df_vix_graph = get_stock_price("^VIX","2018-01-01")
        return page1.layout
    elif pathname == '/dash-graphtrek-technical-analyzer/page2':
        page1.df_vix_graph = get_stock_price("^VIX","2018-01-01")
        return page1.layout
    else:
        return page1.layout

if __name__ == '__main__':
    app.run_server(debug=False)