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


class Label:

    def __init__(self, tickers=None):
        self.tickers = tickers
        self.raw_df = None
        self.label_df = None

    def get_tickers(self, folder='../data/excelfull'):
        '''
        Get all tickers in a folder

        Parameters
        ----------
        folder : str
            Path to folder wanted to get tickers. Ex: '../data/excelfull'

        Returns
        -------
        self : Label
            Label with tickers (list of strings)
        '''
        file_names = pd.Series(os.listdir(folder))
        file_names = file_names.sort_values().str.split('_')
        tickers = [file_name[0] for file_name in file_names]
        tickers = [ticker for ticker in tickers if len(ticker) == 3]
        self.tickers = tickers
        return self

    def get_raw(self, folder='../data/excelfull'):
        '''
        Get raw price data

        Parameters
        ----------
        folder: list of strings
            List of tickers. Ex: ['TCB', 'CTG', 'GAS']

        Returns
        -------
        self : Label
            Label with raw of prices (pandas.DataFrame)
        '''
        paths = []
        for ticker in self.tickers:
            paths.append(f'{folder}/{ticker}_excelfull.csv')

        raw_df = []
        for path in paths:
            raw = pd.read_csv(path)
            raw_df.append(raw)
        raw_df = pd.concat(raw_df).reset_index(drop=True)
        self.raw_df = raw_df
        return self

    @staticmethod
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

    @staticmethod
    def group_dates(input_dates, last_date, term=1, gap=3):
        '''
        Get list of last dates of months in every periods month

        Parameters
        ----------
        input_dates : list or series of str
            Input dates. Ex: ['20210731', '20210730', '20210729', ...]
        last_date : str
            Last date that wanted to group to. Ex: '20210731'
        term : int
            Term in months from last_date. Ex: 1
            Term must not greater than gap.
        gap : int
            Gap in months between periods. Ex: 3

        Returns
        -------
        list of str
            List of last dates. Ex: ['20210731', '20210731', '20210731', ...]
        '''
        # # covert string inputs to integers
        # input_dates = [int(input_date) for input_date in input_dates]
        # last_dates = Label.get_last_dates(last_date, gap=gap)
        # last_dates = [int(last_date) for last_date in last_dates]

        # # group input dates to corresponding last dates
        # # idea: get nearest (larger) last date of every input date
        # subtract = np.subtract.outer(input_dates, last_dates)
        # subtract_neg = np.where(subtract > 0, -np.inf, subtract)
        # subtract_argmax = np.argmax(subtract_neg, axis=1)
        # group_dates = [last_dates[index] for index in subtract_argmax]

        # # some input dates greater than last date, replace them by na
        # dates_greater_last_date = np.array(input_dates) > int(last_date)
        # dates_greater_last_date_index = np.where(dates_greater_last_date)[0]
        # group_dates = [str(group_date) for group_date in group_dates]
        # group_dates = pd.Series(group_dates)
        # group_dates[dates_greater_last_date_index] = np.nan
        # group_dates = list(group_dates)
        # return group_dates

        # covert input dates to datetime64[D]
        input_dates = [datetime.strptime(str(input_date), '%Y%m%d') for input_date in input_dates]
        input_dates = [datetime.strftime(input_date, '%Y-%m-%d') for input_date in input_dates]
        input_dates = np.array(input_dates, dtype='datetime64[D]')

        # covert last dates to datetime64[D]
        last_dates = Label.get_last_dates(last_date, gap=gap)
        last_dates = [datetime.strptime(str(last_date), '%Y%m%d') for last_date in last_dates]
        last_dates = [datetime.strftime(last_date, '%Y-%m-%d') for last_date in last_dates]
        last_dates = np.array(last_dates, dtype='datetime64[D]')

        # calculate gap between last dates and input dates
        # the output is an 2D array whose shape is len(input_dates) x len(last_dates)
        day_from_last_to_input = np.subtract.outer(last_dates, input_dates)
        day_from_last_to_input = np.array(day_from_last_to_input, dtype='int')

        # replace any gap between 0 and 30 days by the corresponding last date
        # otherwise, replace by 0 (later, the array will be summed by axis=0)
        for i, last_date in enumerate(last_dates):
            condition = (0 <= day_from_last_to_input[i]) & (day_from_last_to_input[i] < term * 30)
            day_from_last_to_input[i][condition] = int(str(last_date).replace('-', ''))
            day_from_last_to_input[i][~condition] = 0

        # sum the array
        group_dates = np.sum(day_from_last_to_input, axis=0)
        group_dates = [str(group_date) if group_date != 0 else np.nan for group_date in group_dates]
        return group_dates

    def get_label(
        self,
        last_date='20210731',
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
        self : Label
            Label with final label data (pandas.DataFrame)
            Ex. columns names: ['Ticker', 'Label_Time', 'Feat_Time', 'Return']
        '''
        label_df = (
            self.raw_df
            # Raw input data having ['<Ticker>', '<DTYYYYMMDD>', '<CloseFixed>']
            .rename(columns={'<Ticker>': id_col})

            # create group last dates. Ex: '20210710' -> '20210731'
            .assign(**{label_time_col: lambda df: (
                Label.group_dates(df['<DTYYYYMMDD>'], last_date=last_date, gap=gap))})

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
        self.label_df = label_df
        return self

    def export(self, path='label.csv'):
        self.label_df.to_csv(path, index=False)


if __name__ == '__main__':
    label = Label()
    label.get_tickers()
    label.get_raw()
    label.get_label()
    label.export()
