import os

import pandas as pd
import datetime

from feat_gen import Finance
from feat_params import Params


def get_hours_minutes_seconds(timedelta):
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def get_tickers(folder):
    file_names = pd.Series(os.listdir(folder))
    file_names = file_names.sort_values().str.split('_')
    tickers = [x[0] for x in file_names]
    tickers = [x for x in tickers if len(x) == 3]
    return tickers


tickers = get_tickers('data/reportfinance')
# tickers = ['VSI']
start = datetime.datetime.now()
bs_df = []
for i, ticker in enumerate(tickers):
    finance = Finance(Params)
    finance.params.ticker = ticker
    finance.params.input_path = (
        finance
        .params
        .gen_input_path(finance.params.ticker)
    )
    finance.load_data(**finance.params.__dict__)
    finance.get_bs(**finance.params.__dict__)
    finance.get_bs_qtr(**finance.params.__dict__)
    bs_df.append(finance.bs_qtr)

    end_i = datetime.datetime.now()
    until_i = end_i - start
    est_total = (end_i - start) * len(tickers) / (i + 1)
    est_remain = est_total - until_i
    _hours, minutes, seconds = get_hours_minutes_seconds(est_remain)
    print(f'{i+1:5}/{len(tickers)} \
        ---> Finishing {ticker} \
        ---> Remaining {minutes:2} minutes, {seconds:2} seconds')


print(f' ----------> Done')
bs_df = pd.concat(bs_df).reset_index(drop=True)
bs_df.to_csv(Params.__dict__['output_path'], index=False)
