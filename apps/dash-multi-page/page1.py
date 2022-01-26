import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

def serve_layout():
    return html.Div([
            html.H3('Page 1'),
            dcc.Dropdown(
                id='page1-dropdown',
                options=[
                    {'label': 'Page 1 - {}'.format(i), 'value': i} for i in [
                        'NYC', 'MTL', 'LA'
                    ]
                ]
            ),
            html.Div(id='page1-display-value'),
            dcc.Link('Go to Page 2', href='/page2')
        ])


@app.callback(
    Output('page1-display-value', 'children'),
    [Input('page1-dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)