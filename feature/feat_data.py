'''
This script is to generate features for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-07-31'


import os

import pandas as pd


class Finance:

    ticker_col = 'Ticker'
    time_col = 'Feat_Time'

    def __init__(self, ticker):
        self.ticker = ticker
        self.raw = None

    @staticmethod
    def __gen_raw_path(ticker):
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
        raw_path = Finance.__gen_raw_path(ticker)
        cols = list(range(0, 100))
        raw = pd.read_csv(raw_path, sep='\t', names=cols)
        self.raw = raw


class Balance(Finance):

    start_bs = 'Tài sản ngắn hạn###Current Assets'
    end_bs = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'

    def __init__(self, ticker):
        self.ticker = ticker
        # self.bs_raw = None
        # self.bs_eng = None
        # self.bs_adequate_years = None
        # self.bs_dates = None
        # self.bs_clean = None

    def get_bs(self):
        start_bs = self.start_bs
        end_bs = self.end_bs
        raw = self.raw
        # save colume names and fill NA by an integer; otherwise get float .0
        cols = raw.iloc[0, 1:].fillna(1900).astype(int)
        bs_raw = (
            raw
            .drop(0)  # remove 'CAN DOI KE TOAN' and years rows
            .set_index(0)  # set elements of bs as index (to filter at next step)
            .loc[start_bs: end_bs]  # slice through the index
            .set_axis(cols, axis=1)  # set the column names as pre-determined
            .dropna(axis=1, how='all')  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        bs_raw.index = bs_raw.index.set_names('').astype(str)
        self.bs_raw = bs_raw

    def __get_ith_or_last_elements(self, ls: list, i: int = 2):
        ith_elements = []
        for li in ls:
            if len(li) >= i:
                ith_elements.append(li[i - 1])
            else:
                ith_elements.append(li[-1])
        return ith_elements

    def get_bs_eng(self):
        bs_raw = self.bs_raw
        cols_split = bs_raw.columns.str.split(pat='###')
        eng_cols = self.__get_ith_or_last_elements(cols_split)
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
            .str.replace(' ', '_')
            .str.title()
        )
        eng_cols[24] = 'Owners_Equity_Net'  # duplicates Owners_Equity
        bs_raw.columns = eng_cols
        self.bs_eng = bs_raw

    def __get_years_keep(self, years):
        years_keep = (
            years
            .to_series()
            .groupby(level=0)
            .count()
            .loc[lambda x: (x == 1) | (x == 5) | (x.index == '2021')]
            .index
        )
        return years_keep

    def remove_inadequate_years(self):
        bs_eng = self.bs_eng
        years_keep = self.__get_years_keep(bs_eng.index)
        self.bs_adequate_years = bs_eng.loc[lambda df: df.index.isin(years_keep)]

    def __convert_2021_to_dates(self, years):
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
        else:
            raise Exception('Year 2021 appears more than twice.')
        return dates

    def __convert_years_once_to_dates(self, years):
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

    def __convert_years_five_times_to_dates(self, years):
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

    def convert_years_to_dates(self, remove_years_once=True):
        bs_adequate_years = self.bs_adequate_years
        dates_2021 = self.__convert_2021_to_dates(bs_adequate_years.index)
        dates_years_once = self.__convert_years_once_to_dates(bs_adequate_years.index)
        dates_years_five_times = self.__convert_years_five_times_to_dates(bs_adequate_years.index)

        dates = dates_2021 + dates_years_once + dates_years_five_times
        dates = pd.Series(dates, dtype=str)
        dates = (
            dates
            .sort_values(ascending=False)
            .str.replace('9999', '')
            .reset_index(drop=True)
        )
        bs_dates = bs_adequate_years.set_index(dates)
        if remove_years_once:
            bs_dates = bs_dates.loc[lambda df: [len(date) == 8 for date in df.index.values]]
        self.bs_dates = bs_dates

    def add_ticker(self):
        bs_dates = self.bs_dates
        time_col = self.time_col
        ticker_col = self.ticker_col
        ticker = self.ticker

        bs_dates.index = bs_dates.index.set_names(time_col)
        bs_dates = bs_dates.reset_index()
        bs_dates.insert(0, ticker_col, ticker)
        self.bs_clean = bs_dates


class Income(Finance):

    start_ic = 'Tài sản ngắn hạn###Current Assets'
    end_ic = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'
    ic_path = 'bs.csv'

    def __init__(self, ticker):
        self.ticker = ticker

    def get_ic(self):
        start_ic = self.start_ic
        end_ic = self.end_ic
        raw = self.raw
        '''
        '''
        # self.ic_raw = ic_raw


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
    bs_data = []
    for i, ticker in enumerate(tickers):
        bs = Balance(ticker)
        bs.load_data()
        bs.get_bs()
        bs.get_bs_eng()
        bs.remove_inadequate_years()
        bs.convert_years_to_dates()
        bs.add_ticker()
        bs_data.append(bs.bs_clean)
        # print log
        print(f'{i+1:5}/{len(tickers)} \t ---> Finishing {ticker}')

    print('---> Done')
    bs_data = pd.concat(bs_data).reset_index(drop=True)
    bs_data.to_csv('bs_data.csv', index=False)
