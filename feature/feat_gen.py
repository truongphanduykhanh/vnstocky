import datetime

from dateutil.relativedelta import relativedelta
import pandas as pd

bs = pd.read_csv('bs_data.csv', dtype={'Ticker': str, 'Feat_Time': str})


def get_common_bs(
    bs,
    remove_common=[
        'Non_Current_Assets', 'Tangible_Assets', 'Leased_Assets', 'Intangible_Assets',
        'Provision', 'Other_Liabilities', 'Total_Assets', 'Total_Equity'],
    keep_absolute=[
        'Total_Assets', 'Liabilities', 'Owners_Equity_Net']):

    remove_common = [f'{col}_Common' for col in remove_common]
    keep_absolute_df = bs[keep_absolute]
    meta_df = bs[['Ticker', 'Feat_Time']]
    total_assets = bs['Total_Assets']
    common_cols = [col for col in bs.columns if col not in ['Ticker', 'Feat_Time']]
    common_df = bs[common_cols].divide(total_assets, axis=0)
    common_df = common_df.add_suffix('_Common')
    output_df = pd.concat([meta_df, keep_absolute_df, common_df], axis=1)
    output_df = output_df.loc[:, lambda df: ~df.columns.isin(remove_common)]
    return output_df


def get_last_dates(input_time,
                   delta=1,
                   input_format='%Y%m%d',
                   output_format='%Y%m%d'
                   ):
    '''
    Get list of last dates of months or weeks

    Parameters
    ----------
    input_time : str
        input time. Ex: '20210630'
    delta : int
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
    output_time = input_time + relativedelta(months=delta)
    output_time = output_time.replace(day=28) + relativedelta(days=4)
    output_time = output_time - relativedelta(days=output_time.day)
    output_time = output_time.strftime(output_format)
    return output_time


bs_feat = (
    get_common_bs(bs)
    .assign(Feat_Time=lambda df: df['Feat_Time'].apply(get_last_dates, delta=1))
)

bs_feat.to_csv('bs_feat.csv', index=False)
