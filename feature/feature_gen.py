'''
This script is to generate features data for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-08-22'


import os
import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


class Utility:

    @staticmethod
    def get_ith_or_last_elements(ls: list[list], i: int = 2) -> list:
        '''
        In a list including n sublists, get i-th element of every sublist.
        If i>=len(sublist), get last elemment of every sublist.
        '''
        ith_elements = []
        for li in ls:
            if len(li) >= i:
                ith_elements.append(li[i - 1])
            else:
                ith_elements.append(li[-1])
        return ith_elements

    @staticmethod
    def get_years_keep(years: list[str], remove_years_once=True) -> list[str]:
        '''
        In a list of years, e.g. ['2021', '2020', '2020', '2020', '2020', '2020', '2019'],
        keep only 2021 and the years that appear exactly once (or five times).
        '''
        if remove_years_once:
            years_keep = (
                pd.Series(years)
                .value_counts()
                .loc[lambda x: (x == 5) | (x.index == '2021')]
                .sort_index(ascending=False)
                .index
            )
        else:
            years_keep = (
                pd.Series(years)
                .value_counts()
                .loc[lambda x: (x == 1) | (x == 5) | (x.index == '2021')]
                .sort_index(ascending=False)
                .index
            )
        years_keep = list(years_keep)
        return years_keep

    @staticmethod
    def convert_2021_to_dates(years: list[str]) -> list[str]:
        '''
        In a list of years, e.g. ['2021', '2020', '2020', '2020', '2020', '2020', '2019'],
        return the quarter ending dates of 2021. E.g. ['20210331'].
        '''
        count_2021 = (
            pd.Series(years)
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
    def convert_years_once_to_dates(years: list[str]) -> list[str]:
        '''
        In a list of years, e.g. ['2021', '2020', '2020', '2020', '2020', '2020', '2019'],
        except 2021, add dummy suffix '9999' the years that appear exactly once.
        E.g. ['20129999'].
        '''
        years_once = (
            pd.Series(years)
            .value_counts()
            .loc[lambda x: (x == 1) & (x.index != '2021')]
            .sort_index(ascending=False)
            .index
        )
        years_once = list(years_once)
        dates = [year + '9999' for year in years_once]
        return dates

    @staticmethod
    def convert_years_five_times_to_dates(years: list[str]) -> list[str]:
        '''
        In a list of years, e.g. ['2021', '2020', '2020', '2020', '2020', '2020', '2019'],
        for years that appear exactly five times,
        add dummy suffix '9999' to the first appearance (represent the full year),
        add quarter ending dates to the remaining four appearances.
        '''
        years_five_times = (
            pd.Series(years)
            .value_counts()
            .loc[lambda x: x == 5]
            .sort_index(ascending=False)
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
    def get_last_dates(input_time: str,
                       lag: int = 0,
                       input_format: str = '%Y%m%d',
                       output_format: str = '%Y%m%d'
                       ) -> list[str]:
        '''Get list of last dates of months.'''
        input_time = datetime.datetime.strptime(input_time, input_format)
        output_time = input_time + relativedelta(months=lag)
        output_time = output_time.replace(day=28) + relativedelta(days=4)
        output_time = output_time - relativedelta(days=output_time.day)
        output_time = output_time.strftime(output_format)
        return output_time


class FS:

    ticker_col = 'Ticker'
    time_col = 'Feat_Time'

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.raw = None  # raw data
        self.fs = None  # financial statement (income or balance)

    @staticmethod
    def gen_raw_path(ticker: str) -> str:
        '''Get directory path of raw data.'''
        return f'../data/reportfinance/{ticker}_reportfinance.csv'

    def load_data(self):
        '''
        Load raw data to object.
        '''
        # balance sheet and income statement may have different fields length
        # create a long dummy column names; otherwise, error would arise
        ticker = self.ticker
        raw_path = FS.gen_raw_path(ticker)
        cols = list(range(0, 100))
        raw = pd.read_csv(raw_path, sep='\t', names=cols)
        self.raw = raw

    def get_fs_raw(self):
        '''
        Get income/balance raw from the raw data.
        Need customed construction for income statement and balance sheet.
        '''
        pass

    def clean_fs_column_names(self):
        '''
        Get english columns for income/balance.
        Need customed construction for income statement and balance sheet.
        '''
        pass

    def __remove_inadequate_years(self, remove_years_once: bool = True):
        '''Remove years that don't appear exactly once, except 2021.'''
        years = self.fs.index
        years_keep = Utility.get_years_keep(years, remove_years_once=remove_years_once)
        self.fs = self.fs.loc[lambda df: df.index.isin(years_keep)]

    def convert_feat_time_to_dates(
        self,
        remove_years_once: bool = True,
        remove_years_full: bool = True
    ):
        '''Convert feature time from years to quarter ending dates.'''
        self.__remove_inadequate_years(remove_years_once=remove_years_once)

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
        if remove_years_full:
            self.fs = self.fs.loc[lambda df: [len(date) == 8 for date in df.index.values]]

    def lag_feat_time(self, lag: int = 1):
        '''
        Financial statements are published 1 or 2 months late.
        Lag the feature time to realize the true context.
        '''
        feat_time_lag = self.fs.index.to_series().apply(Utility.get_last_dates, lag=lag)
        self.fs.index = feat_time_lag

    def add_ticker(self):
        '''Add ticker to the financial statement (income or balance).'''
        fs = self.fs
        time_col = self.time_col
        ticker_col = self.ticker_col
        ticker = self.ticker

        fs.index = fs.index.set_names(time_col)
        fs = fs.reset_index()
        fs.insert(0, ticker_col, ticker)
        self.fs = fs


class Income(FS):

    start_income = 'Tổng doanh thu hoạt động kinh doanh###Gross Sale Revenues'
    end_income = 'Lợi nhuận sau thuế thu nhập doanh nghiệp###Profit after Corporate Income Tax'

    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_income_raw(self):
        '''Get income raw from the raw data.'''
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
            .astype(float)  # ticker KDC has error typing of string. need to convert data to float
            .set_axis(cols, axis=1)  # set the column names as pre-determined
            .drop(columns=[1900.0])  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        income.index = income.index.set_names('').astype(str)
        self.fs = income

    def clean_income_column_names(self):
        '''Get english columns for income.'''
        cols_split = self.fs.columns.str.split(pat='###')
        eng_cols = Utility.get_ith_or_last_elements(cols_split)
        eng_cols = [col.title().replace(' ', '_') for col in eng_cols]
        self.fs.columns = eng_cols


class Balance(FS):

    start_balance = 'Tài sản ngắn hạn###Current Assets'
    end_balance = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'

    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_balance_raw(self):
        '''Get balance raw from the raw data.'''
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
            .astype(float)  # ticker KDC has error typing of string. need to convert data to float
            .set_axis(cols, axis=1)  # set the column names as pre-determined
            .drop(columns=[1900.0])  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        balance.index = balance.index.set_names('').astype(str)
        self.fs = balance

    def clean_balance_column_names(self):
        '''Get english columns for balance.'''
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


def get_income(ticker: str) -> pd.DataFrame:
    '''Get income statement for one ticker.'''
    income = Income(ticker)
    income.load_data()
    income.get_income_raw()
    income.clean_income_column_names()
    income.convert_feat_time_to_dates(remove_years_once=True, remove_years_full=True)
    income.lag_feat_time(lag=1)
    income.add_ticker()
    return income.fs


def get_balance(ticker: str) -> pd.DataFrame:
    '''Get balance sheet for one ticker.'''
    balance = Balance(ticker)
    balance.load_data()
    balance.get_balance_raw()
    balance.clean_balance_column_names()
    balance.convert_feat_time_to_dates(remove_years_once=True, remove_years_full=True)
    balance.lag_feat_time(lag=1)
    balance.add_ticker()
    return balance.fs


def get_ratios(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    meta_cols: list[str] = ['Ticker', 'Feat_Time']
) -> pd.DataFrame:
    '''Calulate financial ratios from income and balance.'''
    income_balance = pd.merge(income, balance, on=meta_cols, how='outer')
    meta = income_balance[meta_cols]
    ratios = pd.DataFrame({
        # profitability
        'Gross_Margin_Ratio': income_balance['Gross_Profit'] / income_balance['Net_Sales'],
        'Operating_Margin_Ratio': income_balance['Net_Profit_From_Operating_Activities'] / income_balance['Net_Sales'],
        'Return_On_Equity_Ratio': income_balance['Profit_After_Corporate_Income_Tax'] / income_balance['Owners_Equity'],
        'Return_On_Assets_Ratio': income_balance['Profit_After_Corporate_Income_Tax'] / income_balance['Total_Assets'],

        # liquidity
        'Current_Ratio': income_balance['Current_Assets'] / income_balance['Short_Term_Liabilities'],
        'Quick_Ratio': (income_balance['Current_Assets'] - income_balance['Inventory']) / income_balance['Short_Term_Liabilities'],
        'Cash_Ratio': income_balance['Cash_And_Cash_Equivalents'] / income_balance['Short_Term_Liabilities'],

        # leverage
        'Debt_Ratio': income_balance['Liabilities'] / income_balance['Total_Assets'],
        'Debt_To_Equity_Ratio': income_balance['Liabilities'] / income_balance['Owners_Equity'],
        'Interest_Coverage_Ratio': income_balance['Net_Profit_From_Operating_Activities'] / income_balance['Other_Expenses'],

        # efficiency
        'Assets_Turnover_Ratio': income_balance['Net_Sales'] / income_balance['Total_Assets'],
        'Inventory_Turnover_Ratio': income_balance['Cost_Of_Goods_Sold'] / income_balance['Inventory'],
        'Receivables_Turnover_Ratio': income_balance['Net_Sales'] / income_balance['Short_Term_Account_Receivables'],
        'Days_Sales_Inventory_Ratio': 365 / (income_balance['Cost_Of_Goods_Sold'] / income_balance['Inventory'])
    })
    ratios = pd.concat([meta, ratios], axis=1)
    return ratios


class FSFeatures:

    @staticmethod
    def calculate_roll_mean(
        df: pd.DataFrame,
        window: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate moving average (roll mean) of financial statement.'''
        meta = df[meta_cols]
        roll_mean = (
            df
            .drop(meta_cols[1], axis=1)
            .groupby(meta_cols[0])
            .rolling(window)
            .mean()
            .shift(-window + 1)
            .reset_index(drop=True)
        )
        roll_mean = roll_mean.add_suffix(f'_Mean_{window}Q')
        roll_mean = pd.concat([meta, roll_mean], axis=1)
        return roll_mean

    @staticmethod
    def shift_data(
        df: pd.DataFrame,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Shift financial statement periods quarters back to history.'''
        meta = df[meta_cols]
        shift = (
            df
            .drop(meta_cols[1], axis=1)
            .groupby(meta_cols[0])
            .shift(-periods)
            .reset_index(drop=True)
        )
        shift = pd.concat([meta, shift], axis=1)
        return shift

    @staticmethod
    def calculate_momentum(
        df: pd.DataFrame,
        window: int,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate the growth rate (momentum) of financial statement'''
        meta = df[meta_cols]

        roll_mean = FSFeatures.calculate_roll_mean(df, window)
        shift = FSFeatures.shift_data(roll_mean, periods)

        roll_mean = roll_mean.drop(meta_cols, axis=1)
        shift = shift.drop(meta_cols, axis=1)

        momentum = roll_mean.div(shift)
        momentum = momentum.add_suffix(f'_Momen_{periods}Q')
        momentum = pd.concat([meta, momentum], axis=1)
        return momentum

    @staticmethod
    def calculate_momentum_loop(
        df: pd.DataFrame,
        window_list: list[int] = [1, 2, 4],
        periods_list: list[int] = [1, 2, 4],
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calulate growth rate (momentum) over different windows and periods settings.'''
        meta = df[meta_cols]
        roll_mean_momentum = []
        for window in window_list:
            for periods in periods_list:
                momentum_window_periods = FSFeatures.calculate_momentum(df, window, periods)
                momentum_window_periods = momentum_window_periods.drop(meta_cols, axis=1)
                roll_mean_momentum.append(momentum_window_periods)
        roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
        return roll_mean_momentum

    @staticmethod
    def get_common_size(
        df: pd.DataFrame,
        master_col: str,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ):
        '''Calulate common size of financial statement.'''
        df_common = (
            df
            .set_index(meta_cols)
            .divide(df.set_index(meta_cols)[master_col], axis=0)
            .add_suffix('_Common')
            .reset_index()
        )
        return df_common

    @staticmethod
    def calculate_momentum_common(
        df: pd.DataFrame,
        window: int,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate the growth rate (momentum) of financial statement common size'''
        meta = df[meta_cols]
        roll_mean = FSFeatures.calculate_roll_mean(df, window)
        shift = FSFeatures.shift_data(roll_mean, periods)
        roll_mean = roll_mean.drop(meta_cols, axis=1)
        shift = shift.drop(meta_cols, axis=1)

        momentum = roll_mean.subtract(shift)
        momentum = momentum.add_suffix(f'_Momen_{periods}Q')
        momentum = pd.concat([meta, momentum], axis=1)
        return momentum

    @staticmethod
    def calculate_momentum_common_loop(
        df: pd.DataFrame,
        window_list: list[int] = [1, 2, 4],
        periods_list: list[int] = [1, 2, 4],
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''
        Calulate growth rate (momentum) of financial statement common size
        over different windows and periods settings.
        '''
        meta = df[meta_cols]
        roll_mean_momentum = []
        for window in window_list:
            for periods in periods_list:
                momentum_window_periods = FSFeatures.calculate_momentum_common(df, window, periods)
                momentum_window_periods = momentum_window_periods.drop(meta_cols, axis=1)
                roll_mean_momentum.append(momentum_window_periods)
        roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
        return roll_mean_momentum


def get_tickers(folder: str = '../data/excelfull') -> str:
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

    from functools import reduce

    tickers = get_tickers('../data/excelfull')

    # INCOME STATMENT
    income = []
    for i, ticker in enumerate(tickers):
        income_i = get_income(ticker)
        income.append(income_i)
        print(f'{i+1:5}/{len(tickers)} \t ---> Finish Income {ticker}')

    print('---> Finish Income')
    income = pd.concat(income).reset_index(drop=True)
    income_momen = FSFeatures.calculate_momentum_loop(income)
    income_common = FSFeatures.get_common_size(income, 'Gross_Sale_Revenues')
    income_common_momen = FSFeatures.calculate_momentum_common_loop(income_common)

    # BALANCE SHEET
    balance = []
    for i, ticker in enumerate(tickers):
        balance_i = get_balance(ticker)
        balance.append(balance_i)
        print(f'{i+1:5}/{len(tickers)} \t ---> Finish Balance {ticker}')

    print('---> Finish Balance')
    balance = pd.concat(balance).reset_index(drop=True)
    balance_momen = FSFeatures.calculate_momentum_loop(balance)
    balance_common = FSFeatures.get_common_size(balance, 'Total_Assets')
    balance_common_momen = FSFeatures.calculate_momentum_common_loop(balance_common)

    # RATIOS
    ratios = get_ratios(income, balance)
    ratios_momen = FSFeatures.calculate_momentum_common_loop(ratios)

    feature_list = [
        income, income_momen, income_common, income_common_momen,
        balance, balance_momen, balance_common, balance_common_momen,
        ratios, ratios_momen]

    feature = reduce(
        lambda left, right: pd.merge(
            left, right, on=['Ticker', 'Feat_Time'], how='outer'), feature_list)

    feature.to_csv('feature.csv', index=False)
