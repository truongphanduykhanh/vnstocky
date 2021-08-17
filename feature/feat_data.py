'''
This script is to generate features for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-08-17'


import os
import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


class Utility:

    @staticmethod
    def get_ith_or_last_elements(ls: list, i: int = 2):
        ith_elements = []
        for li in ls:
            if len(li) >= i:
                ith_elements.append(li[i - 1])
            else:
                ith_elements.append(li[-1])
        return ith_elements

    @staticmethod
    def get_years_keep(years):
        years_keep = (
            years
            .to_series()
            .groupby(level=0)
            .count()
            .loc[lambda x: (x == 1) | (x == 5) | (x.index == '2021')]
            .index
        )
        return years_keep

    @staticmethod
    def convert_2021_to_dates(years):
        count_2021 = (
            years
            .to_series()
            .loc[lambda x: x == '2021']
            .shape[0]
        )
        if count_2021 == 0:
            dates = []
        elif count_2021 == 1:
            dates = ['20210331']
        elif count_2021 == 2:
            dates = ['20210331', '20210630']
        elif count_2021 == 3:
            dates = ['20210331', '20210630', '20219999']
        else:
            raise Exception('Year 2021 appears more than three times.')
        return dates

    @staticmethod
    def convert_years_once_to_dates(years):
        years_once = (
            years
            .to_series()
            .groupby(level=0)
            .count()
            .loc[lambda x: (x == 1) & (x.index != '2021')]
            .index
        )
        dates = [year + '9999' for year in years_once]
        return dates

    @staticmethod
    def convert_years_five_times_to_dates(years):
        years_five_times = (
            years
            .to_series()
            .groupby(level=0)
            .count()
            .loc[lambda x: x == 5]
            .index
        )
        dates = []
        quarter_ends = ['1231', '0930', '0630', '0331']
        for year in years_five_times:
            dates.append(year + '9999')
            for quarter_end in quarter_ends:
                dates.append(year + quarter_end)
        return dates

    @staticmethod
    def get_last_dates(input_time,
                       lag=1,
                       input_format='%Y%m%d',
                       output_format='%Y%m%d'
                       ):
        '''
        Get list of last dates of months or weeks

        Parameters
        ----------
        input_time : str
            input time. Ex: '20210630'
        lag : int
            number of month wanted to look future. Ex: 1
        input_format : str
            format of input time. Ex: '%Y%m' or '%Y%m%d'
        output_format : str
            format of output time. Ex: '%Y%m' or '%Y%m%d'

        Returns
        -------
        list of str
            last date of next (delta) month. Ex: '20210731'
        '''
        input_time = datetime.datetime.strptime(input_time, input_format)
        output_time = input_time + relativedelta(months=lag)
        output_time = output_time.replace(day=28) + relativedelta(days=4)
        output_time = output_time - relativedelta(days=output_time.day)
        output_time = output_time.strftime(output_format)
        return output_time


class Finance:

    ticker_col = 'Ticker'
    time_col = 'Feat_Time'

    def __init__(self, ticker):
        self.ticker = ticker
        self.raw = None
        self.fs = None

    @staticmethod
    def gen_raw_path(ticker):
        '''
        Get directory path of raw data

        Parameters
        ----------
        ticker : str
            Ticker. Ex: 'TCB

        Returns
        -------
        str
            Directory path of raw data
        '''
        return f'../data/reportfinance/{ticker}_reportfinance.csv'

    def load_data(self):
        # balance sheet and income statement may have different fields length
        # create a long dummy column names; otherwise, error would arise
        ticker = self.ticker
        raw_path = Finance.gen_raw_path(ticker)
        cols = list(range(0, 100))
        raw = pd.read_csv(raw_path, sep='\t', names=cols)
        self.raw = raw

    def get_fs_raw(self):
        pass

    def clean_fs_column_names(self):
        pass

    def __remove_inadequate_years(self):
        years = self.fs.index
        years_keep = Utility.get_years_keep(years)
        self.fs = self.fs.loc[lambda df: df.index.isin(years_keep)]

    def convert_feat_time_to_dates(self, remove_years_once=True):
        self.__remove_inadequate_years()

        dates_2021 = Utility.convert_2021_to_dates(self.fs.index)
        dates_years_once = Utility.convert_years_once_to_dates(self.fs.index)
        dates_years_five_times = Utility.convert_years_five_times_to_dates(self.fs.index)

        dates = dates_2021 + dates_years_once + dates_years_five_times
        dates = pd.Series(dates, dtype=str)
        dates = (
            dates
            .sort_values(ascending=False)
            .str.replace('9999', '')
            .reset_index(drop=True)
        )
        self.fs = self.fs.set_index(dates)
        if remove_years_once:
            self.fs = self.fs.loc[lambda df: [len(date) == 8 for date in df.index.values]]

    def lag_feat_time(self, lag=1):
        feat_time_lag = self.fs.index.to_series().apply(Utility.get_last_dates, lag=lag)
        self.fs.index = feat_time_lag

    def add_ticker(self):
        fs = self.fs
        time_col = self.time_col
        ticker_col = self.ticker_col
        ticker = self.ticker

        fs.index = fs.index.set_names(time_col)
        fs = fs.reset_index()
        fs.insert(0, ticker_col, ticker)
        self.fs = fs


class Income(Finance):

    start_income = 'Tổng doanh thu hoạt động kinh doanh###Gross Sale Revenues'
    end_income = 'Lợi nhuận sau thuế thu nhập doanh nghiệp###Profit after Corporate Income Tax'

    def __init__(self, ticker):
        self.ticker = ticker

    def get_income_raw(self):
        start_income = self.start_income
        end_income = self.end_income
        raw = self.raw
        # save colume names and fill NA by an dummy integer; otherwise get float .0
        cols = raw.loc[raw[0] == 'KET QUA KINH DOANH', 1:].iloc[0]
        cols = pd.to_numeric(cols, errors='coerce').fillna(1900).astype(int)
        income = (
            raw
            .set_index(0)  # set elements of financial statements as index (to filter at next step)
            .loc[lambda df: ~df.index.duplicated(keep='first')]  # there are FS elements duplicates
            .loc[start_income: end_income]  # slice through the index
            .set_axis(cols, axis=1)  # set the column names as pre-determined
            .drop(columns=[1900.0])  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        income.index = income.index.set_names('').astype(str)
        self.fs = income

    def clean_income_column_names(self):
        cols_split = self.fs.columns.str.split(pat='###')
        eng_cols = Utility.get_ith_or_last_elements(cols_split)
        eng_cols = [col.title().replace(' ', '_') for col in eng_cols]
        self.fs.columns = eng_cols


class Balance(Finance):

    start_balance = 'Tài sản ngắn hạn###Current Assets'
    end_balance = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'

    def __init__(self, ticker):
        self.ticker = ticker

    def get_balance_raw(self):
        start_balance = self.start_balance
        end_balance = self.end_balance
        raw = self.raw
        # save colume names and fill NA by an dummy integer; otherwise get float .0
        cols = raw.loc[raw[0] == 'CAN DOI KE TOAN', 1:].iloc[0]
        cols = pd.to_numeric(cols, errors='coerce').fillna(1900).astype(int)
        balance = (
            raw
            .set_index(0)  # set elements of financial statements as index (to filter at next step)
            .loc[lambda df: ~df.index.duplicated(keep='last')]  # there are FS elements duplicates
            .loc[start_balance: end_balance]  # slice through the index
            .set_axis(cols, axis=1)  # set the column names as pre-determined
            .drop(columns=[1900.0])  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        balance.index = balance.index.set_names('').astype(str)
        self.fs = balance

    def clean_balance_column_names(self):
        cols_split = self.fs.columns.str.split(pat='###')
        eng_cols = Utility.get_ith_or_last_elements(cols_split)
        eng_cols = (
            pd.Series(eng_cols)
            .replace('Cash and Cash Euivalents', 'Cash and Cash Equivalents')
            .replace('Tài sản cố định hữu hình - Giá trị hao mòn lũy kế', 'Tangible Assets')
            .replace('Tài sản cố định thuê tài chính - Giá trị hao mòn lũy kế', 'Leased Assets')
            .replace('Tài sản cố định vô hình - Giá trị hao mòn lũy kế', 'Intangible Assets')
            .replace('Lợi thế thương mại', 'Goodwill')
            .replace('Dự phòng nghiệp vụ', 'Provision')
            .replace('No khac', 'Other Liabilities')
            .replace('Lợi ích của cổ đông thiểu số', 'Minority Interest')
            .str.replace('-', '_')
        )
        eng_cols = [col.title().replace(' ', '_') for col in eng_cols]
        self.fs.columns = eng_cols


def get_income(ticker):
    income = Income(ticker)
    income.load_data()
    income.get_income_raw()
    income.clean_income_column_names()
    income.convert_feat_time_to_dates()
    income.lag_feat_time()
    income.add_ticker()
    return income.fs


def get_balance(ticker):
    balance = Balance(ticker)
    balance.load_data()
    balance.get_balance_raw()
    balance.clean_balance_column_names()
    balance.convert_feat_time_to_dates()
    balance.lag_feat_time()
    balance.add_ticker()
    return balance.fs


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


if __name__ == '__main__':

    tickers = get_tickers('../data/excelfull')

    # INCOME STATMENT
    income_data = []
    for i, ticker in enumerate(tickers):
        income = get_income(ticker)
        income_data.append(income)
        # print log
        print(f'{i+1:5}/{len(tickers)} \t ---> Finishing Income {ticker}')

    print('---> Done')
    income_data = pd.concat(income_data).reset_index(drop=True)
    income_data.to_csv('income_data.csv', index=False)

    # BALANCE SHEET
    balance_data = []
    for i, ticker in enumerate(tickers):
        balance = get_balance(ticker)
        balance_data.append(balance)
        # print log
        print(f'{i+1:5}/{len(tickers)} \t ---> Finishing Balance {ticker}')

    print('---> Done')
    balance_data = pd.concat(balance_data).reset_index(drop=True)
    balance_data.to_csv('balance_data.csv', index=False)
