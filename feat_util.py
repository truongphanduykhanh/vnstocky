import pandas as pd


def get_bs(df):
    cols = df.iloc[0, 1:].fillna(1900).astype(int)
    start_bs = 'Tài sản ngắn hạn###Current Assets'
    end_bs = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'
    bs = (
        df
        .drop(0)
        .set_index(0)
        .loc[start_bs: end_bs]
        .set_axis(cols, axis=1)
        .dropna(axis=1, how='all')
        .transpose()
    )
    bs.index.name = 'Feat_Month'
    return bs


def _convert_suffix_to_mth(suffix):
    if str(suffix) == '1':
        mth = '12'
    if str(suffix) == '2':
        mth = '09'
    if str(suffix) == '3':
        mth = '06'
    if str(suffix) == '4':
        mth = '03'
    return mth


def change_index_month(df):
    months = (
        df
        .groupby(level=0)
        .cumcount()
        .astype(str)
        .replace('0', '')
        .apply(_convert_suffix_to_mth)
    )
    index_month = df.index.str.replace('2021', '202103') + months
    return df.set_index(index_month)


def _get_yr_qtr_cols(df):
    cols = df.index.to_list()
    yr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 4)]
    qtr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 6)]
    return yr_cols, qtr_cols


def get_qtr_cols(df):
    _yr_cols, qtr_cols = _get_yr_qtr_cols(df)
    df_qtr = df.loc[qtr_cols, :]
    return df_qtr
