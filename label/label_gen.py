'''
This script is to generate label for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-07-25'

import os
from datetime import datetime

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd


def get_tickers(folder):
    '''
    Get all tickers in a folder

    Parameters
    ----------
    folder : str
        Path to folder wanted to get tickers. Ex: '../data/excelfull'

    Returns
    -------
    list of str
        List of tickers. Ex: ['A32', 'AAM', 'AAT', ...]
    '''
    file_names = pd.Series(os.listdir(folder))
    file_names = file_names.sort_values().str.split('_')
    tickers = [file_name[0] for file_name in file_names]
    tickers = [ticker for ticker in tickers if len(ticker) == 3]
    return tickers


def get_raw(tickers):
    '''
    Get raw price data

    Parameters:
    ----------
    tickers: list of strings
        List of tickers. Ex: ['TCB', 'CTG', 'GAS']

    Returns:
    -------
    pandas.DataFrame
        Data frame of raw prices
    '''
    paths = []
    for ticker in tickers:
        paths.append(f'../data/excelfull/{ticker}_excelfull.csv')

    prices = []
    for path in paths:
        price = pd.read_csv(path)
        prices.append(price)
    prices = pd.concat(prices).reset_index(drop=True)
    return prices


def get_last_dates(
    last_date,
    gap=3,
    periods=100,
    input_format='%Y%m%d',
    out_format='%Y%m%d'
):
    '''
    Get list of last dates of months in every periods month

    Parameters
    ----------
    last_date : str
        Input last date. Ex: '20210731'
    gap : int
        Gap in months between periods. Ex: 3
    periods : int
        Number of periods wanted to look back. Ex: 100
    input_format : str
        Format of input date. Ex: '%Y%m%d'
    output_format : str
        Format of output dates. Ex: '%Y%m%d'

    Returns
    -------
    list of str
        List of last dates. Ex: ['20210731', '20210430', '20210131', ...]
    '''
    last_date = datetime.strptime(str(last_date), input_format)
    dates = []
    for i in range(periods):
        date = last_date - relativedelta(months=i * gap)
        date = date.replace(day=28) + relativedelta(days=4)
        date = date - relativedelta(days=date.day)
        date = date.strftime(out_format)
        dates.append(date)
    return dates


def group_dates(input_dates, last_date, gap=3):
    '''
    Get list of last dates of months in every periods month

    Parameters
    ----------
    input_dates : list or series of str
        Input dates. Ex: ['20210731', '20210730', '20210729', ...]
    last_date : str
        Last date that wanted to group to. Ex: '20210731'
    gap : int
        Gap in months between periods. Ex: 3

    Returns
    -------
    list of str
        List of last dates. Ex: ['20210731', '20210731', '20210731', ...]
    '''
    # covert string inputs to integers
    input_dates = [int(input_date) for input_date in input_dates]
    last_dates = get_last_dates(last_date, gap=gap)
    last_dates = [int(last_date) for last_date in last_dates]
    # group input dates to corresponding last dates
    # idea: get nearest (larger) last date of every input date
    res = np.subtract.outer(input_dates, last_dates)
    res_neg = np.where(res > 0, -np.inf, res)
    res_argmax = np.argmax(res_neg, axis=1)
    res_dates = [last_dates[index] for index in res_argmax]
    return res_dates


def get_label(
    raw_df,
    last_date,
    gap=3,
    id_col='Ticker',
    label_time_col='Label_Time',
    feat_time_col='Feat_Time',
    label_col='Return'
):
    '''
    Get label data frame from raw data.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw input data having ['<Ticker>', '<DTYYYYMMDD>', '<CloseFixed>']
    last_date : str
        Last date that wanted to group to. Ex: '20210731'
    gap : int
        Gap in months between periods. Ex: 3
    id_col : str
        Column name of tickers. Ex: 'Ticker'
    label_time_col : str
        Column name of label time. Ex: 'Label_Time'
    feat_time_col : str
        Column name of feature time. Ex: 'Feat_Time'
    label_col : str
        Column name of label. Ex: 'Return'

    Returns
    -------
    pandas.DataFrame
        Data frame of label. Ex. of column names:
        ['Ticker', 'Label_Time', 'Feat_Time', 'Return']
    '''
    label_df = (
        raw_df
        .rename(columns={'<Ticker>': id_col})
        # create group last dates. Ex: '20210710' -> '20210731'
        .assign(**{label_time_col: lambda df: (
            group_dates(df['<DTYYYYMMDD>'], last_date=last_date, gap=gap))})
        # calculate mean closing prices
        .groupby([id_col, label_time_col])
        .agg({'<CloseFixed>': 'mean'})
        .reset_index()
        .rename(columns={'<CloseFixed>': 'MeanCloseFixed'})
        # add lag prices columne (denominator column)
        .sort_values([id_col, label_time_col], ascending=[True, False])
        .assign(MeanCloseFixedLag=lambda df: (
            df.groupby(id_col)['MeanCloseFixed'].shift(-1)))
        # calculate return
        .assign(**{label_col: lambda df: (
            df['MeanCloseFixed'] / df['MeanCloseFixedLag'] - 1)})
        # add feature time column for later reference
        .assign(**{feat_time_col: lambda df: (
            df.groupby(id_col)[label_time_col].shift(-1))})
        .loc[:, [id_col, label_time_col, feat_time_col, label_col]]
        .dropna()
        # clean data frame
        .astype({label_time_col: int, feat_time_col: int})
        .astype({label_time_col: str, feat_time_col: str})
        .reset_index(drop=True)
    )
    return label_df


if __name__ == '__main__':
    tickers = get_tickers('../data/excelfull')
    raw_df = get_raw(tickers)
    label_df = get_label(raw_df, last_date='20210731', gap=3)
    label_df.to_csv('label.csv', index=False)
