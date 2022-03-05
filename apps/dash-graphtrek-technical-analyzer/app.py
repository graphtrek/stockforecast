import dash
import logging

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], suppress_callback_exceptions=True)
app.logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s %(message)s"))
server = app.server
