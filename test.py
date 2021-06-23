import pandas as pd
import feat_util


class Finance():

    def __init__(self):
        self.input = None
        self.bs = None
        self.bs_qtr = None

    def load_data(self, path='data/reportfinance/TN1_reportfinance.csv'):
        # balance sheet and income statement may have different fields length
        # create a long dummy columns name; otherwise, error would arise
        cols = list(range(0, 100))
        input = pd.read_csv(path, sep='\t', names=cols)
        self.input = input

    def get_bs(self):
        bs = feat_util.get_bs(self.input)
        self.bs = bs

    def get_bs_qtr(self):
        month_index_df = feat_util.change_index_month(self.bs)
        bs_qtr = feat_util.get_qtr_cols(month_index_df)
        self.bs_qtr = bs_qtr
