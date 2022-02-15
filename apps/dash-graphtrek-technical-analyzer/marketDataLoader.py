import schedule
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from symbols import symbols
import time


def get_symbols_info_df():
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

        options['spread%'] = np.round(100 - ((options['bid'] / options['ask']) * 100), 1)
        options.to_csv("/home/nexys/graphtrek/stock/" + ticker.ticker + "_options.csv", index=False)
        print('Get options', ticker.ticker, 'done.')
    except:
        print('Get options', ticker.ticker, 'error.')
    return options


schedule.every(1).minutes.until("22:00").do(get_symbols_info_df)

while True:
    schedule.run_pending()
    time.sleep(1)
