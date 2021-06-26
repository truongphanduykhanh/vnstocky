import pandas as pd
from datetime import timedelta


def _convert_suffix_to_mth(suffix):
    mth = ''
    if str(suffix) == '1':
        mth = '12'
    if str(suffix) == '2':
        mth = '09'
    if str(suffix) == '3':
        mth = '06'
    if str(suffix) == '4':
        mth = '03'
    return mth


def change_index_mth(df):
    mths = (
        df
        .groupby(level=0)
        .cumcount()
        .astype(str)
        .replace('0', '')
        .apply(_convert_suffix_to_mth)
    )
    index_mth = df.index.str.replace('2021', '202103') + mths
    return df.set_index(index_mth)


def _get_yr_qtr_cols(df):
    cols = df.index.to_list()
    yr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 4)]
    qtr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 6)]
    return yr_cols, qtr_cols


def _add_mths(mths, no_mths=1, fmt='%Y%m'):
    mths = pd.Series(mths)
    mths = pd.to_datetime(mths, format=fmt) + timedelta(days=30 * (no_mths + 0.5))
    mths = mths.dt.to_period('M').dt.strftime(fmt)
    return mths


def get_qtr_cols(df):
    _yr_cols, qtr_cols = _get_yr_qtr_cols(df)
    df_qtr = df.loc[qtr_cols, :]
    df_qtr.index = _add_mths(df_qtr.index)
    return df_qtr


def _get_ith_or_last_elements(ls: list, i: int = 2):
    ith_elements = []
    for li in ls:
        if len(li) >= i:
            ith_elements.append(li[i - 1])
        else:
            ith_elements.append(li[-1])
    return ith_elements


def get_eng_cols(df, pat='###'):
    cols_split = df.columns.str.split(pat)
    eng_cols = _get_ith_or_last_elements(cols_split)
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
    df.columns = eng_cols
    return df
