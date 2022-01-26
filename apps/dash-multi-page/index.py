import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
import page1, page2


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page1':
        return page1.layout
    elif pathname == '/page2':
        return page2.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)