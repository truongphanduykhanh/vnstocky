import pandas as pd
import feat_util


class Finance:

    def __init__(self, params):
        self.params = params
        self.input = None
        self.bs = None
        self.bs_qtr = None

    def load_data(self, path, **kwrgs):
        # balance sheet and income statement may have different fields length
        # create a long dummy columns name; otherwise, error would arise
        cols = list(range(0, 100))
        input = pd.read_csv(path, sep='\t', names=cols)
        self.input = input

    def get_bs(self, start_bs, end_bs, elements_col, **kwargs):
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
        bs.index = bs.index.astype(str)
        bs.index.name = elements_col
        self.bs = bs

    def get_bs_qtr(self, **kwargs):
        month_index_df = feat_util.change_index_mth(self.bs)
        bs_qtr = feat_util.get_qtr_cols(month_index_df)
        bs_qtr = feat_util.get_eng_cols(bs_qtr)
        # bs_qtr.index = bs_qtr.index.set_names('Feat_Month')
        # bs_qtr = bs_qtr.reset_index()
        # bs_qtr.insert(0, 'Ticker', 'TN1')
        self.bs_qtr = bs_qtr
