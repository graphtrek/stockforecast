from dash import dcc
from dash import html
import yfinance as yf

from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.momentum import RSIIndicator
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import os.path
import json
from dash_iconify import DashIconify
from app import app
import time

twelve_months = date.today() + relativedelta(months=-12)
eleven_months = date.today() + relativedelta(months=-11)
six_months = date.today() + relativedelta(months=-6)
five_months = date.today() + relativedelta(months=-5)
three_months = date.today() + relativedelta(months=-3)
two_months = date.today() + relativedelta(months=-2)
one_month = date.today() + relativedelta(months=+1)
fourteen_days = date.today() + relativedelta(days=+14)
start_date = six_months.strftime("%Y-%m-%d")
end_date = fourteen_days.strftime("%Y-%m-%d")


def Header(app):
    return html.Div([get_header(app)])
    # return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            # html.Div(
            #     [
            #         html.A(
            #             html.Img(
            #                 src=app.get_asset_url("dash-financial-logo.png"),
            #                 className="logo",
            #             ),
            #             href="https://plotly.com/dash",
            #         ),
            #         html.A(
            #             html.Button(
            #                 "Enterprise Demo",
            #                 id="learn-more-button",
            #                 style={"margin-left": "-10px"},
            #             ),
            #             href="https://plotly.com/get-demo/",
            #         ),
            #         html.A(
            #             html.Button("Source Code", id="learn-more-button"),
            #             href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-financial-report",
            #         ),
            #     ],
            #     className="row",
            # ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H5(id="symbol_name", children=["Graphtrek Technical Analyzer"])
                        ], className="seven columns main-title",
                    ),
                    html.Div(id="symbol", className="five columns")
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/dash-graphtrek-technical-analyzer/page1",
                className="tab first",
            ),
            dcc.Link(
                "Prediction",
                href="/dash-graphtrek-technical-analyzer/page2",
                className="tab",
            )
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]], style={"text-align": "left"}))
        table.append(html.Tr(html_row))
    return table


def make_options_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    if df is not None and len(df) > 0:
        df = df.rename(columns={'openInterest': 'O.I.', 'impliedVolatility': 'I.V.', 'expirationDate': 'EXP.DATE'},
                       inplace=False)
        html_header_row = []
        for header in np.array(df.columns):
            html_header_row.append(html.Th([header.upper()]))
        table.append(html.Tr(html_header_row))
        df['O.I.'] = df['O.I.'].fillna(0)  # replace all NaN values with zeros
        df['O.I.'] = df['O.I.'].astype(int)  # convert column from float to integer
        df['I.V.'] = df['I.V.'].fillna(0)  # replace all NaN values with zeros
        df['I.V.'] = df['I.V.'].astype(int)  # convert column from float to integer
        df['volume'] = df['volume'].fillna(0)  # replace all NaN values with zeros
        df['volume'] = df['volume'].astype(int)  # convert column from float to integer

        max_open_interest = df["O.I."].idxmax()
        max_volume = df["volume"].idxmax()
        for index, row in df.iterrows():
            html_row = []
            for i in range(len(row)):
                td_val = row[i]
                if i == 4 or i == 5:
                    td_val = '{:,}'.format(td_val)
                html_row.append(html.Td([td_val]))

            if index == max_open_interest:
                table.append(html.Tr(html_row, className="oi-max-text"))
            elif index == max_volume:
                table.append(html.Tr(html_row, className="vol-max-text"))
            else:
                table.append(html.Tr(html_row))
    return table


def indicators(df):
    # MACD
    macd = MACD(close=df['Close'],
                window_slow=26,
                window_fast=12,
                window_sign=9)
    # stochastics
    stoch = StochasticOscillator(high=df['High'],
                                 close=df['Close'],
                                 low=df['Low'],
                                 window=14,
                                 smooth_window=3)

    rsi = RSIIndicator(close=df['Close'], window=14)
    return macd, stoch, rsi


def is_support(df, i):
    cond1 = df['Low'][i] < df['Low'][i - 1]
    cond2 = df['Low'][i] < df['Low'][i + 1]
    cond3 = df['Low'][i + 1] < df['Low'][i + 2]
    cond4 = df['Low'][i - 1] < df['Low'][i - 2]
    return cond1 and cond2 and cond3 and cond4


def is_resistance(df, i):
    cond1 = df['High'][i] > df['High'][i - 1]
    cond2 = df['High'][i] > df['High'][i + 1]
    cond3 = df['High'][i + 1] > df['High'][i + 2]
    cond4 = df['High'][i - 1] > df['High'][i - 2]
    return cond1 and cond2 and cond3 and cond4


def is_far_from_level(value, levels, df):
    ave = np.mean(df['High'] - df['Low'])
    return np.sum([abs(value - level) < ave for level in levels]) == 0


def find_nearest_greater_than(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff < 0] = np.inf
    idx = diff.argmin()
    return inputData[idx]


def find_nearest_less_than(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff > 0] = -np.inf
    idx = diff.argmax()
    return inputData[idx]


def get_stock_price(ticker, from_date):
    df = yf.download(ticker.ticker, start=from_date, interval="1d")
    # df = df.rename(columns={"Close": "Close1", "Adj Close": "Close"})

    # ticker = yf.Ticker(symbol)

    # df = ticker.history(start=from_date, interval="1d")
    # print(df.info())
    df['Date'] = pd.to_datetime(df.index)
    # df['Date'] = df['Date'].apply(mpl_dates.date2num)
    # df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['MA300'] = df['Close'].rolling(window=300).mean()

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi().to_numpy()

    # macd, soch, rsi = indicators(df)
    # df['RSI'] = rsi.rsi().to_numpy()
    # df['MACD_DIFF'] = macd.macd_diff().to_numpy()
    # df['MACD'] = macd.macd().to_numpy()
    # df['MACD_SIGNAL'] = macd.macd_signal().to_numpy()
    app.logger.info('Get Stock Price %s %s', ticker.ticker, 'done.')
    return df


def get_stock_price1(ticker, from_date):
    file_path = '/home/nexys/graphtrek/stock/' + ticker.ticker + '.csv'
    file_exists = os.path.exists(file_path)
    if file_exists is True and not is_file_older_than(file_path, 300):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        app.logger.info('Get CACHED Stock Price %s %s', ticker.ticker, 'done.')
    else:
        df = get_stock_price(ticker, from_date)
        df.to_csv(file_path, index=True)
    return df


def is_file_older_than(file, secs=1):
    file_time = os.path.getmtime(file)
    # Check against 24 hours
    return (time.time() - file_time) > secs


def calculate_levels(chart_df):
    levels = []
    low = 0
    high = np.round(chart_df['High'].max(), 1)
    close_price, last_date, prev_close_price = get_last_price(chart_df)
    for i in range(2, len(chart_df) - 2):
        try:
            if is_support(chart_df, i):
                low = chart_df['Low'][i]
            if is_far_from_level(low, levels, chart_df):
                levels.append(low)
            elif is_resistance(chart_df, i):
                high = chart_df['High'][i]
            if is_far_from_level(high, levels, chart_df):
                levels.append(high)
        except:
            app.logger.error('calculate_levels error')

    levels.append(np.max(levels) * 1.02)

    levels = sorted(levels, reverse=True)

    min_level = np.round(find_nearest_less_than(close_price * 0.99, levels), 1)
    # min_level = np.round(find_nearest_less_than(min_level_0 * 0.99, levels), 1)
    if min_level > close_price:
        min_level = np.round(close_price, 1)

    max_level = np.round(find_nearest_greater_than(close_price * 1.01, levels), 1)
    # max_level = np.round(find_nearest_greater_than(max_level_0 * 1.01, levels), 1)
    if max_level < close_price:
        max_level = np.round(close_price, 1)

    # app.logger.info('Calculate Levels close_price %s %s %s %s %4',
    #   close_price, 'min_level:', min_level, 'max_level:', max_level)
    return levels, close_price, min_level, max_level


def load_options_df(symbol):
    file_path = "/home/nexys/graphtrek/stock/" + symbol + "_options.csv"
    file_exists = os.path.exists(file_path)
    if file_exists is True:
        options_df = pd.read_csv(file_path)
        options_df['impliedVolatility'] = np.round(options_df['impliedVolatility'] * 100, 2)
        options_df['percentChange'] = np.round(options_df['percentChange'], 2)
        options_df = options_df.drop(
            columns=['contractSize',
                     'currency',
                     'change',
                     'percentChange',
                     'lastTradeDate',
                     'lastPrice',
                     'inTheMoney',
                     'contractSymbol'])
        # options_df['strike'] = options_df['strike'].astype(float).round(decimals=1)
        # options_df['strike'].round(decimals=1)
        return options_df
    return None


def find_level_option_interests(options_df, min_level, max_level, dte_min, dte_max):
    if options_df is not None:
        new_header = options_df.columns  # grab the first row for the header
        put_query = \
            "CALL == False" \
            " and strike>" + str(int(min_level)) + \
            " and strike<=" + str(int(max_level)) + \
            " and dte>" + str(dte_min) + \
            " and dte<=" + str(dte_max)
        PUT_options_df = options_df.query(put_query)

        if len(PUT_options_df) > 1:
            put_max_openInterest_index = PUT_options_df["openInterest"].idxmax()
            # put_max_volume_index = PUT_options_df["volume"].idxmax()
            PUT_options_to_return_df = PUT_options_df.loc[put_max_openInterest_index:put_max_openInterest_index]
            # if put_max_volume_index != put_max_openInterest_index:
            #    PUT_options_to_return_df = \
            #        PUT_options_to_return_df.append(PUT_options_df.loc[put_max_volume_index:put_max_volume_index])
            PUT_options_to_return_df.columns = new_header  # set the header row as the df header
            # app.logger.info("PUT options found: %s %s %s", str(len(PUT_options_df)) + " query:", put_query)
        else:
            app.logger.error("No PUT options found query: %s", put_query)
            PUT_options_to_return_df = pd.DataFrame(columns=new_header)

        PUT_options_to_return_df = PUT_options_to_return_df.drop(columns=['CALL'])
        call_query = \
            "CALL == True" + \
            " and strike>" + str(min_level) + \
            " and strike<" + str(max_level) + \
            " and dte>" + str(dte_min) + \
            " and dte<" + str(dte_max)
        CALL_options_df = options_df.query(call_query)

        if len(CALL_options_df) > 1:
            call_max_openInterest_index = CALL_options_df["openInterest"].idxmax()
            # call_max_volume_index = CALL_options_df["volume"].idxmax()
            CALL_options_to_return_df = CALL_options_df.loc[call_max_openInterest_index:call_max_openInterest_index]
            # if call_max_volume_index != call_max_openInterest_index:
            #    CALL_options_to_return_df = CALL_options_to_return_df.append(CALL_options_df.loc[call_max_volume_index:call_max_volume_index])
            CALL_options_to_return_df.columns = new_header  # set the header row as the df header
            # app.logger.info("CALL options found: %s %s %s", str(len(CALL_options_df)) + " query:", call_query)
        else:
            app.logger.info("No CALL options found query: %s", put_query)
            CALL_options_to_return_df = pd.DataFrame(columns=new_header)

        CALL_options_to_return_df = CALL_options_to_return_df.drop(columns=['CALL'])

        return PUT_options_to_return_df, CALL_options_to_return_df
    app.logger.info("options df is empty")
    return None, None


def get_text(prefix, value, suffix):
    if value is not None and value:
        return prefix + str(value) + suffix
    return ""


def get_info_dict(symbol):
    file_path = "/home/nexys/graphtrek/stock/" + symbol + "_info.json"
    file_exists = os.path.exists(file_path)
    if file_exists is True:
        f = open(file_path, "r")
        info_dict = json.loads(f.read())
        f.close()
        return info_dict
    return None


def get_symbols_info_df(symbols):
    symbols_df = pd.DataFrame(symbols, columns=['Symbol'])
    earnings_list = []
    info_list = []
    quote_type = "EQUITY"
    for symbol in symbols_df["Symbol"]:
        ticker_sector = ""
        ticker_industry = ""
        ticker_short_ratio = ""
        recommendation_key = ""
        short_name = ""
        target_price = ""
        info_file_path = "/home/nexys/graphtrek/stock/" + symbol + "_info.json"
        info_file_exists = os.path.exists(info_file_path)
        if info_file_exists is True:
            ticker_info = get_info_dict(symbol)
            if ticker_info is not None:
                quote_type = ticker_info['quoteType']
                if 'sector' in ticker_info:
                    ticker_sector = ticker_info['sector']
                if 'industry' in ticker_info:
                    ticker_industry = ticker_info['industry']
                if 'shortRatio' in ticker_info:
                    ticker_short_ratio = str(ticker_info['shortRatio'])
                if 'recommendationKey' in ticker_info:
                    recommendation_key = str(ticker_info['recommendationKey'])
                if 'shortName' in ticker_info:
                    short_name = str(ticker_info['shortName'])
                if 'targetMedianPrice' in ticker_info:
                    target_price = str(ticker_info['targetMedianPrice'])

        info_list.append([ticker_sector, ticker_industry, ticker_short_ratio, recommendation_key, short_name, target_price])

        earning_date_str = ""
        nr_of_days = None
        if quote_type == "EQUITY":
            calendar_file_path = "/home/nexys/graphtrek/stock/" + symbol + "_calendar.csv"
            calendar_file_exists = os.path.exists(calendar_file_path)
            if calendar_file_exists is True:
                calendar_df = pd.read_csv(calendar_file_path)
                if calendar_df is not None:
                    earning_date_str = ""
                    nr_of_days = None
                    try:
                        earning_datetime_str = calendar_df.iloc[0][1]
                        earning_date_str = earning_datetime_str[0:10]
                        earning_date = datetime.strptime(earning_date_str, '%Y-%m-%d')
                        days = earning_date - datetime.today()
                        if days.days >= 0:
                            nr_of_days = days.days
                    except:
                        None

        earnings_list.append([earning_date_str, nr_of_days])

    earnings_array = np.array(earnings_list)
    symbols_df['Earning'], symbols_df['Day'] = earnings_array[:, 0], earnings_array[:, 1]
    info_array = np.array(info_list)

    symbols_df['Sector'], \
        symbols_df['Industry'],\
        symbols_df['ShortRatio'], \
        symbols_df['Recommendation'], \
        symbols_df['Name'], \
        symbols_df['TargetPrice'] = \
        info_array[:, 0], \
        info_array[:, 1], \
        info_array[:, 2], \
        info_array[:, 3],\
        info_array[:, 4], \
        info_array[:, 5]

    # print(symbols_df)
    return symbols_df


def get_last_price(df):
    last_day_df = df.iloc[-1:]
    before_last_day_df = df.tail(2)
    last_date = last_day_df['Date'].dt.strftime('%Y-%m-%d')
    close_price = np.round(float(df.iloc[-1:]['Adj Close']), 2)
    prev_close_price = np.round(float(before_last_day_df['Adj Close'].iloc[0]), 2)
    # print(close_price, prev_close_price)
    return close_price, last_date, prev_close_price


def get_title(ticker, df, predict_price):
    close_price, last_date, prev_close_price = get_last_price(df)

    ath = np.round(float(df['Adj Close'].max()))
    discount = np.round(ath - close_price, 1)
    discount_percent = np.round((discount / ath) * 100, 2)

    title = ticker.ticker + " " + last_date \
        + " Last Price:$" + str(close_price) \
        + " ($" + str(predict_price) + " in 2 weeks)" \
        + " Discount:$" + str(discount) + " (" + str(discount_percent) + "%)"
    return html.A(title, href='https://in.tradingview.com/chart?symbol=' + ticker.ticker, target="_blank")


def get_predict_price(close_price, first_prediction, mean_prediction, max_level, min_level):
    predict_price = np.round((close_price / first_prediction) * mean_prediction, 2)
    if predict_price > max_level:
        predict_price = max_level
    if predict_price < min_level:
        predict_price = min_level
    app.logger.info("predict_price: %s", predict_price)
    return predict_price


def display_chart(ticker, df):
    close_price, last_date, prev_close_price = get_last_price(df)
    # zoom_df = df.iloc['Date' >= start_date]
    # zoom_df = df[df.Date >= six_months.strftime("%Y-%m-%d")]
    # y_zoom_max = zoom_df["High"].max()
    # y_zoom_min = zoom_df["Low"].min()
    # range_start_date = six_months.strftime("%Y-%m-%d")
    # range_end_date = date.today().strftime("%Y-%m-%d")

    fig = go.Figure(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        line={"color": "#97151c"},
        mode="lines",
        name=ticker.ticker + " $" + str(close_price)
    ))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    fig.update_layout(
        autosize=True,
        #        width=700,
        height=200,
        font={"family": "Raleway", "size": 10},
        margin={
            "r": 30,
            "t": 30,
            "b": 30,
            "l": 30,
        },
        showlegend=True,
        dragmode='pan',
        titlefont={
            "family": "Raleway",
            "size": 10,
        },
        xaxis={
            # "autorange": True,
            # "range": [
            #    range_start_date,
            #    range_end_date
            #],
            "rangeselector": {
                "buttons": [
                    {
                        "count": 7,
                        "label": "1W",
                        "step": "day",
                        "stepmode": "backward"
                    },
                    {
                        "count": 1,
                        "label": "1M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 3,
                        "label": "3M",
                        "step": "month",
                        "stepmode": "backward"

                    },
                    {
                        "count": 6,
                        "label": "6M",
                        "step": "month",
                        "stepmode": "backward"
                    },
                    {
                        "count": 1,
                        "label": "1Y",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "label": "All",
                        "step": "all",
                    },
                ]
            },
            "showline": True,
            "type": "date",
            "zeroline": False
        },
        yaxis={
            "autorange": True,
            "showline": True,
            "type": "linear",
            "zeroline": False,
        }
    )

    fig.add_trace(go.Scatter(
        x=[np.min(df['Date']), np.max(df['Date'])],
        y=[close_price, close_price],
        mode="lines",
        line=dict(shape='linear', color='rgb(10, 120, 24)', dash='dot'),
        name='Last Price:' + ticker.ticker + ' $' + str(close_price)
    ))

    if ticker.ticker == "^VIX":
        # fig.update_xaxes(type="date", range=[three_months, date.today()])
        fig.update_xaxes(type="date",
                         range=[three_months, date.today()],
                         showline=True,
                         linewidth=1,
                         linecolor='#ccc',
                         mirror=True,
                         showgrid=True,
                         gridwidth=1,
                         gridcolor='LightPink',
                         showspikes=True,
                         spikemode='across',
                         spikesnap='cursor',
                         spikedash='dash')

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[30, 30],
            mode="lines",
            line=dict(shape='linear', color='rgb(255, 0, 0)'),
            name='Panic +$30'
        ))

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[25, 25],
            mode="lines",
            line=dict(shape='linear', color='rgb(255, 187, 0)'),
            name='Fear +$25'
        ))

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[22.5, 22.5],
            mode="lines",
            line=dict(shape='linear', color='rgb(0, 255, 42)'),
            name='Hedge +$22.5'
        ))
    else:
        # fig.update_xaxes(type="date", range=[twelve_months, date.today()])
        fig.update_xaxes(type="date",
                         range=[six_months, date.today()],
                         showline=True,
                         linewidth=1,
                         linecolor='#ccc',
                         mirror=True,
                         showgrid=True,
                         gridwidth=1,
                         gridcolor='LightPink',
                         showspikes=True,
                         spikemode='across',
                         spikesnap='cursor',
                         spikedash='dash')

        ath = np.round(float(df['Close'].max()))
        pullback_level = np.round((ath * 0.95), 1)
        correction_level = np.round((ath * 0.9), 1)
        crash_level = np.round((ath * 0.8), 1)

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[pullback_level, pullback_level],
            mode="lines",
            line=dict(shape='linear', color='rgb(0, 255, 42)'),
            name='Pullback $' + str(pullback_level)
        ))

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[correction_level, correction_level],
            mode="lines",
            line=dict(shape='linear', color='rgb(255, 187, 0)'),
            name='Correction $' + str(correction_level)
        ))

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[crash_level, crash_level],
            mode="lines",
            line=dict(shape='linear', color='rgb(255, 0, 0)'),
            name='Crash $' + str(crash_level)
        ))

    # fig.update_yaxes(showline=True, linewidth=1, linecolor='#ccc', mirror=True)

    fig.update_layout(xaxis_rangeslider_visible=False)
    # fig.update_xaxes(type="date", range=[range_start_date, range_end_date])
    # fig.update_yaxes(range=[y_zoom_min, y_zoom_max])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    # fig.update_xaxes(showgrid=True,  gridwidth=1, gridcolor='LightPink', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash')
    return fig


def get_predictions(symbol):
    if os.path.exists("/home/nexys/graphtrek/stock/" + symbol + "_test_prediction.csv") and \
            os.path.exists("/home/nexys/graphtrek/stock/" + symbol + "_prediction.csv"):
        indicators_test_prediction_df = pd.read_csv("/home/nexys/graphtrek/stock/" + symbol + "_test_prediction.csv")
        indicators_prediction_df = pd.read_csv("/home/nexys/graphtrek/stock/" + symbol + "_prediction.csv")
        return indicators_test_prediction_df, indicators_prediction_df
    return None, None


def display_analyzer(symbol, df, indicators_test_prediction_df, indicators_prediction_df):
    levels, close_price, min_level, max_level = calculate_levels(df)

    zoom_df = df[df.Date >= twelve_months.strftime("%Y-%m-%d")]

    zoom_df1 = df[df.Date >= two_months.strftime("%Y-%m-%d")]
    y_zoom_max = zoom_df1["High"].max()
    y_zoom_min = zoom_df1["Low"].min() * 0.999

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.012,
                        row_heights=[0.70, 0.10, 0.20],
                        specs=[
                            [{"type": "candlestick"}],
                            [{"type": "bar"}],
                            [{"type": "scatter"}]
                        ])

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    fig.update_xaxes(showline=True, linewidth=1, linecolor='#ccc', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#ccc', mirror=True)

    fig.add_trace(go.Candlestick(x=zoom_df['Date'],
                                 open=zoom_df['Open'],
                                 high=zoom_df['High'],
                                 low=zoom_df['Low'],
                                 close=zoom_df['Close'],
                                 name=symbol + " $" + str(close_price),
                                 showlegend=True), row=1, col=1)

    fig.update_layout(
        autosize=True,
        #        width=700,
        height=800,
        font={"family": "Raleway", "size": 10},
        margin={
            "r": 10,
            "t": 10,
            "b": 10,
            "l": 10,
        },
        showlegend=True,
        dragmode='pan',
        titlefont={
            "family": "Raleway",
            "size": 10,
        },
        xaxis={
            # "autorange": True,
            # "range": [
            #     start_date,
            #     end_date
            # ],
            # "rangeselector": {
            #     "buttons": [
            #         # {
            #         #     "count": 1,
            #         #     "label": "1M",
            #         #     "step": "month",
            #         #     "stepmode": "backward"
            #         # },
            #         {
            #             "count": 3,
            #             "label": "3M",
            #             "step": "month",
            #             "stepmode": "backward"
            #
            #         },
            #         {
            #             "count": 6,
            #             "label": "6M",
            #             "step": "month",
            #             "stepmode": "backward"
            #         },
            #         {
            #             "count": 1,
            #             "label": "1Y",
            #             "step": "year",
            #             "stepmode": "backward",
            #         },
            #         {
            #             "label": "All",
            #             "step": "all",
            #         },
            #     ]
            # },
            "showline": True,
            "type": "date",
            "zeroline": True
        },
         yaxis={
        #    "autorange": True,
        #     "showline": True,
        #     "type": "linear",
        #     "zeroline": False,
            }
    )

    # add moving average traces
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['MA20'],
                             line=dict(color='lightgreen', width=2),
                             fill=None,
                             mode='lines',
                             name='MA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['EMA21'],
                             fill='tonexty',
                             mode='lines',
                             line=dict(color='green', width=2),
                             name='EMA 21'), row=1, col=1)
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['MA50'],
                             line=dict(color='blue', width=2),
                             name='MA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['MA100'],
                             line=dict(color='orange', width=2),
                             name='MA 100'), row=1, col=1)
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['MA200'],
                             line=dict(color='red', width=2),
                             name='MA 200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['MA300'],
                             line=dict(color='black', width=2),
                             name='MA 300'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[np.min(df['Date']), np.max(df['Date'])],
        y=[close_price, close_price],
        mode="lines+text",
        line=dict(shape='linear', color='rgb(10, 120, 24)', dash='dot'),
        textfont=dict(size=16, color='black', family='Arial, sans-serif'),
        name='Last Price:' + symbol + ' $' + str(close_price),
        showlegend=True,
        text=['', ' $' + str(close_price) + " (Last Price)", ''],
        textposition="top left"
    ), row=1, col=1)

    ath_percent = 0
    if levels is not None:
        for idx, level in enumerate(levels):
            percent = 0
            if idx == 0:
                ath = level
            current_level = level
            if current_level == 0:
                current_level = 1
            if idx > 0:
                prev_level = levels[idx - 1]
                diff = prev_level - current_level
                percent = (diff / current_level) * 100

                ath_diff = ath - current_level
                ath_percent = (ath_diff / ath) * 100

            if level <= (min_level * 0.99) or level >= (max_level * 1.01):
                line_color = 'rgba(100, 10, 100, 0.2)'
                line_fill = None
                line_width = 1
            else:
                line_color = 'rgba(51, 102, 153, 1)'
                line_fill = None  # 'tonexty'
                line_width = 2
                fig.add_trace(go.Scatter(
                    x=[df['Date'].min(), df['Date'].max()],
                    y=[level, level],
                    mode="lines+text",
                    name="Levels",
                    fill=line_fill,
                    showlegend=False,
                    text=['', '$' + str(np.round(current_level, 1)) + ' (' + str(np.round(percent, 1)) + '% disc:' + str(
                        np.round(ath_percent, 1)) + '%)', ''],
                    textposition="top right",
                    line=dict(shape='linear', color=line_color, dash='dash', width=line_width)
                ), row=1, col=1)

    # Volume
    colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=zoom_df['Date'], y=zoom_df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=zoom_df['Date'],
                             y=zoom_df['RSI'],
                             line=dict(color='royalblue', width=2),
                             name='RSI(14)'
                             ), row=3, col=1)

    if indicators_prediction_df is not None and indicators_test_prediction_df is not None:
        fig.add_trace(go.Scatter(x=indicators_test_prediction_df['Date'],
                                 y=indicators_test_prediction_df['Prediction'],
                                 line=dict(color='firebrick', width=3, dash='dash'),
                                 name='RSI(14) Test Predict'
                                 ), row=3, col=1)

        fig.add_trace(go.Scatter(x=indicators_prediction_df.head(14)['Date'],
                                 y=indicators_prediction_df.head(14)['Prediction'],
                                 line=dict(color='firebrick', width=3, dash='dot'),
                                 name='RSI(14) Future Predict'
                                 ), row=3, col=1)

        first_prediction = np.round(indicators_prediction_df['Prediction'][0])
        mean_prediction = np.round(np.mean(indicators_prediction_df['Prediction']))
        app.logger.info("first_prediction: %s %s %s", first_prediction, "mean_prediction:", mean_prediction)
        predict_price = get_predict_price(close_price,first_prediction,mean_prediction, max_level, min_level)

        fig.add_trace(go.Scatter(
            x=[np.min(df['Date']), np.max(df['Date'])],
            y=[predict_price, predict_price],
            mode="lines+text",
            line=dict(shape='linear', color='crimson', dash='dot'),
            textfont=dict(size=16, color='black', family='Arial, sans-serif'),
            name='Predict Price:' + symbol + ' $' + str(predict_price),
            showlegend=True,
            text=['', ' $' + str(predict_price) + " (Predict Price)", ''],
            textposition="top left"
        ), row=1, col=1)


        fig.add_trace(go.Scatter(
            x=[np.min(indicators_prediction_df['Date']), np.max(indicators_prediction_df.head(14)['Date'])],
            y=[first_prediction, mean_prediction],
            mode="lines",
            line=dict(shape='linear', color='rgb(255, 153, 0)'),
            name='RSI(14) Mean Predict'
        ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[np.min(zoom_df['Date']), np.max(zoom_df['Date'])],
        y=[30, 30],
        mode="lines",
        line=dict(shape='linear', color='rgb(10, 120, 24)', dash='dash'),
        name='RSI(14) Over Sold'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[np.min(zoom_df['Date']), np.max(zoom_df['Date'])],
        y=[50, 50],
        mode="lines",
        line=dict(shape='linear', color='rgb(10, 12, 240)', dash='dash'),
        name='RSI(14) Neutral'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[np.min(zoom_df['Date']), np.max(zoom_df['Date'])],
        y=[70, 70],
        mode="lines",
        line=dict(shape='linear', color='rgb(100, 10, 100)', dash='dash'),
        name='RSI(14) Over Bought'
    ), row=3, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(type="date", range=[start_date, end_date])
    fig.update_yaxes(range=[y_zoom_min, y_zoom_max], row=1, col=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikedash='dash')
    fig.update_xaxes(showspikes=True, spikemode='across', spikedash='dash')

    return fig
