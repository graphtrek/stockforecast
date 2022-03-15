import schedule
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from symbols import symbols
import time
import json
from app import app


def get_symbols_info():
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info_dict = ticker.info
        if info_dict is not None:
            try:
                # create json object from dictionary
                json_info_dict = json.dumps(info_dict)
                # open file for writing, "w"
                f = open("/home/nexys/graphtrek/stock/" + ticker.ticker + "_info.json", "w")
                # write json object to file
                f.write(json_info_dict)
                # close file
                f.close()
                app.logger.info('Get info %s %s', ticker.ticker, 'done.')
            except Exception as e:
                app.logger.info('Get info %s %s', ticker.ticker, 'error.', e)


def get_symbols_options_chain():
    app.logger.info("get_symbols_info_df running")
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        options_chain(ticker)


def options_chain(ticker):
    # tk = yf.Ticker(symbol)
    # Expiration dates
    exps = ticker.options

    # Get options for each expiration
    options = pd.DataFrame()
    try:
        for e in exps:
            opt = ticker.option_chain(e)
            opt = pd.DataFrame().append(opt.calls).append(opt.puts)
            opt['expirationDate'] = e
            options = options.append(opt, ignore_index=True)

        if len(options) > 1:
            # Bizarre error in yfinance that gives the wrong expiration date
            # Add 1 day to get the correct expiration date
            options['expirationDate'] = pd.to_datetime(options['expirationDate'])
            options.insert(0, 'dte', (options['expirationDate'] - datetime.today()).dt.days + 1)
            options['expirationDate'] = options['expirationDate'].dt.date
            # Boolean column if the option is a CALL x : True if (x > 10 and x < 20) else False
            options.insert(1, "CALL", options['contractSymbol'].str[4:].apply(lambda x: "C" in x))

            options[['bid',
                     'ask',
                     'strike',
                     'lastPrice',
                     'volume',
                     'change',
                     'percentChange',
                     'openInterest',
                     'impliedVolatility']] = options[[
                                                    'bid',
                                                    'ask',
                                                    'strike',
                                                    'lastPrice',
                                                    'volume',
                                                    'change',
                                                    'percentChange',
                                                    'openInterest',
                                                    'impliedVolatility']].apply(pd.to_numeric)

            # replace all NaN values with zeros
            options['bid'] = options['bid'].fillna(0)
            options['ask'] = options['ask'].fillna(0)
            options['strike'] = options['strike'].fillna(0)
            options['volume'] = options['volume'].fillna(0)
            options['change'] = options['change'].fillna(0)
            options['percentChange'] = options['percentChange'].fillna(0)
            options['openInterest'] = options['openInterest'].fillna(0)
            options['impliedVolatility'] = options['impliedVolatility'].fillna(0)

            # convert column from float to integer
            options['change'] = options['change'].astype(int)
            options['percentChange'] = options['percentChange'].astype(int)
            options['volume'] = options['volume'].astype(int)
            options['openInterest'] = options['openInterest'].astype(int)
            options['impliedVolatility'] = options['impliedVolatility'].astype(int)
            options['spread%'] = np.round(100 - ((options['bid'] / options['ask']) * 100), 1)

            options.to_csv("/home/nexys/graphtrek/stock/" + ticker.ticker + "_options.csv", index=False)
            app.logger.info('Get options %s %s', ticker.ticker, 'done.')
        else:
            app.logger.info('Get options %s %s', ticker.ticker, 'blank.')
    except e:
        app.logger.info('Get options %s %s %s', ticker.ticker, 'error.', e)
    return None


def schedule_options():
    schedule.every(1).minutes.until("21:50").do(get_symbols_options_chain)
    while True:
        schedule.run_pending()
        time.sleep(1)


get_symbols_info()


# ticker = yf.Ticker("TQQQ")
# options_chain(ticker)


#schedule_options()
