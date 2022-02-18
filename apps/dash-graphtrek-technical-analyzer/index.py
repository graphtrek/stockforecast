from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
import page1
from symbols import symbols
from utils import (
    get_symbols_info_df,
)


def serve_layout():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])


app.layout = serve_layout


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    page1.symbols_info_df = get_symbols_info_df(symbols)
    if pathname == '/dash-graphtrek-technical-analyzer/page1':
        return page1.layout
    else:
        return page1.layout


if __name__ == '__main__':
    app.run_server(debug=False)