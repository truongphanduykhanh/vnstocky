import pandas as pd

import feat_util


class Finance:

    def __init__(self, params):
        self.params = params
        self.input = None
        self.bs = None
        self.bs_qtr = None

    def load_data(self, input_path, **kwrgs):
        # balance sheet and income statement may have different fields length
        # create a long dummy column names; otherwise, error would arise
        cols = list(range(0, 100))
        input = pd.read_csv(input_path, sep='\t', names=cols)
        self.input = input

    def get_bs(self, start_bs, end_bs, **kwargs):
        # save colume names and fill NA by an integer; otherwise get float .0
        cols = self.input.iloc[0, 1:].fillna(1900).astype(int)
        bs = (
            self.input
            .drop(0)  # remove 'CAN DOI KE TOAN' and years rows
            .set_index(0)  # set elements of bs as index (to filter at next step)
            .loc[start_bs: end_bs]  # slice through the index
            .set_axis(cols, axis=1)  # set the column names as pre-dedermined
            .dropna(axis=1, how='all')  # drop redundant cols when add dummy cols
            .transpose()  # set years/quarters as index
        )
        bs.index = bs.index.set_names('').astype(str)
        self.bs = bs

    def get_bs_qtr(self, time_col, ticker_col, ticker, **kwargs):
        month_index_df = feat_util.change_index_mth(self.bs)
        bs_qtr = feat_util.get_qtr_cols(month_index_df)
        bs_qtr = feat_util.get_eng_cols(bs_qtr)
        bs_qtr.index = bs_qtr.index.set_names(time_col)
        bs_qtr = bs_qtr.reset_index()
        bs_qtr.insert(0, ticker_col, ticker)
        self.bs_qtr = bs_qtr

    def export_data(self, output_path, **kwargs):
        self.bs_qtr.to_csv(output_path, index=False)


class FinanceRate:

    def __init__(self) -> None:
        pass
