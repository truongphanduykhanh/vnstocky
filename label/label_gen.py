from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd


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
        paths.append(f'data/excelfull/{ticker}_excelfull.csv')

    prices = []
    for path in paths:
        price = pd.read_csv(path)
        prices.append(price)
    prices = pd.concat(prices).reset_index(drop=True)
    return prices


def get_last_dates(input_date, periods=40, gap=3, input_format='%Y%m%d', out_format='%Y%m%d'):
    '''
    Get list of last dates of months in every periods month

    Parameters
    ----------
    input_date : str
        Input date. Ex: '20210731'
    periods : int
        Number of periods wanted to look back. Ex: 40
    gap : int
        Gap in months between periods. Ex: 3
    input_format : str
        Format of input date. Ex: '%Y%m%d'
    output_format : str
        Format of output date. Ex: '%Y%m%d'

    Returns
    -------
    list of str
        List of last dates. Ex: ['20210731', '20210430', '20210131']
    '''
    input_date = datetime.strptime(str(input_date), input_format)
    dates = []
    for i in range(periods):
        date = input_date - relativedelta(months=i * gap)
        date = date.replace(day=28) + relativedelta(days=4)
        date = date - relativedelta(days=date.day)
        date = date.strftime(out_format)
        dates.append(date)
    return dates


def put_months_to_groups(months, groups, look_back=3, format='%Y%m'):
    groups.sort()
    months_to_groups = pd.Series([np.nan] * len(months))
    for group in groups:
        gap = pd.to_datetime(group, format=format) - pd.to_datetime(months, format=format)
        gap = gap.dt.days
        within_look_back = [0 <= x <= 30 * (look_back - 0.5) for x in gap]
        months_to_groups[within_look_back] = months_to_groups[within_look_back].fillna(group)
    return months_to_groups


def get_feat_months(label_months, format='%Y%m', look_back=3):
    feat_months = pd.to_datetime(label_months, format=format) - timedelta(days=30 * (look_back - 0.5))
    feat_months = feat_months.dt.to_period('M').dt.strftime(format)
    return feat_months


label_months = get_months('200001', '202201', period=3)
feat_months = get_months('199910', '202110', period=3)


label = (
    prices
    .assign(Trading_Month=lambda df: df['<DTYYYYMMDD>'].astype(str).str.slice(0, 6))
    .assign(Label_Month=lambda df: put_months_to_groups(df['Trading_Month'], label_months, look_back=3))
    .groupby(['<Ticker>', 'Label_Month'])
    .agg({'<CloseFixed>': 'mean'})
    .reset_index()
    .assign(Feat_Month=lambda df: get_feat_months(df['Label_Month']))
    .rename(columns={'<Ticker>': 'Ticker', '<CloseFixed>': 'Label_Price'})
)


feat = (
    prices
    .assign(Trading_Month=lambda df: df['<DTYYYYMMDD>'].astype(str).str.slice(0, 6))
    .assign(Feat_Month=lambda df: put_months_to_groups(df['Trading_Month'], feat_months, look_back=3))
    .groupby(['<Ticker>', 'Feat_Month'])
    .agg({'<CloseFixed>': 'mean'})
    .reset_index()
    .rename(columns={'<Ticker>': 'Ticker', '<CloseFixed>': 'Feat_Price'})
)


label_final = (
    label
    .merge(feat, how='left', on=['Ticker', 'Feat_Month'])
    .assign(Return=lambda df: df['Label_Price'] / df['Feat_Price'] - 1)
    .loc[:, ['Ticker', 'Label_Month', 'Feat_Month', 'Return']]
)

label_final.to_csv('label.csv', index=False)
