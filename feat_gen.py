import pandas as pd
import numpy as np
from datetime import timedelta


finance = pd.read_csv('data/reportfinance/TCB_reportfinance.csv', sep='\t')
# finance.info()
# finance.head()


def get_bs(finance):
    start_bs = 'Tài sản ngắn hạn###Current Assets'
    end_bs = 'TỔNG CỘNG NGUỒN VỐN ###TOTAL EQUITY'
    bs = (
        finance
        .rename(columns={' 2021': '2021.4'})
        .set_index('CAN DOI KE TOAN')
        .loc[start_bs: end_bs]
        .reset_index()
    )
    return bs


def get_inc(finance):
    start_inc = 'Tổng doanh thu hoạt động kinh doanh###Gross Sale Revenues'
    end_inc = 'Lợi nhuận sau thuế thu nhập doanh nghiệp###Profit after Corporate Income Tax'
    inc = (
        finance
        .rename(columns={' 2021': '2021.4'})
        .set_index('CAN DOI KE TOAN')
        .loc[start_inc: end_inc]
        .reset_index()
    )
    return inc


def get_eng_element(df):
    df = df.rename(columns={'CAN DOI KE TOAN': 'Elements'})
    element_split = df['Elements'].str.split('###')
    eng_element = []
    for x in element_split:
        if len(x) == 1:
            eng_element.append(x[0])
        if len(x) == 2:
            eng_element.append(x[1])

    eng_element = (
        pd.Series(eng_element)
        .replace('Cash and Cash Euivalents', 'Cash and Cash Equivalents')
        .replace('Tài sản cố định hữu hình - Giá trị hao mòn lũy kế', 'Tangible Assets')
        .replace('Tài sản cố định thuê tài chính - Giá trị hao mòn lũy kế', 'Leased Assets')
        .replace('Tài sản cố định vô hình - Giá trị hao mòn lũy kế', 'Intangible Assets')
        .replace('Lợi thế thương mại', 'Goodwill')
        .replace('Dự phòng nghiệp vụ', 'Provision')
        .replace('No khac', 'Other Liabilities')
        .replace('Lợi ích của cổ đông thiểu số', 'Minority Interest')
        .str.title()
        .str.replace('-', '_')
        .str.replace(' ', '_')
    )
    df['Elements'] = eng_element
    return df


def _get_yr_qtr_cols(df):
    cols = df.columns.str.strip()
    yr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 4)]
    qtr_cols = pd.Series(cols)[lambda s: s.map(lambda x: len(x) == 6)]
    return yr_cols, qtr_cols


def get_qtr_cols(df):
    df.columns = df.columns.str.strip()
    yr_cols, qtr_cols = _get_yr_qtr_cols(df)
    df_qtr = df[['Elements'] + qtr_cols.to_list()]
    return df_qtr


def get_yr_cols(df):
    df.columns = df.columns.str.strip()
    yr_cols, qtr_cols = _get_yr_qtr_cols(df)
    df_yr = df[['Elements'] + yr_cols.to_list()]
    return df_yr


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


def _add_mths(mths, no_mths=1, fmt='%Y%m'):
    mths = pd.to_datetime(mths, format=fmt) + timedelta(days=30 * (no_mths + 0.5))
    mths = mths.dt.to_period('M').dt.strftime(fmt)
    return mths


def change_suffix_to_mths(bs_qtr):
    cols_suffix = pd.Series(bs_qtr.columns[1:])
    yrs = cols_suffix.str[0:4]
    mths = cols_suffix.str[5].apply(_convert_suffix_to_mth)
    cols_months = _add_mths(yrs + mths, 1)
    bs_qtr.columns = ['Elements'] + list(cols_months)
    return bs_qtr


bs = get_bs(finance)
bs_eng = get_eng_element(bs)
bs_qtr = get_qtr_cols(bs_eng)
bs_mth = change_suffix_to_mths(bs_qtr)
bs_yr = get_yr_cols(bs_eng)

inc = get_inc(finance)
inc_eng = get_eng_element(inc)
inc_qtr = get_qtr_cols(inc_eng)
inc_mth = change_suffix_to_mths(inc_qtr)
inc_yr = get_yr_cols(inc_eng)
