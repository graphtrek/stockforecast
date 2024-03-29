import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H3('Page 2'),
    dcc.Dropdown(
        id='page2-dropdown',
        options=[
            {'label': 'Page 2 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='page2-display-value'),
    dcc.Link('Go to Page 1', href='/page1')
])


@app.callback(
    Output('page2-display-value', 'children'),
    [Input('page2-dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)