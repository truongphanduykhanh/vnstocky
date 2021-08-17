import datetime

from dateutil.relativedelta import relativedelta
import pandas as pd

income = pd.read_csv('income_data.csv', dtype={'Ticker': str, 'Feat_Time': str})
balance = pd.read_csv('balance_data.csv', dtype={'Ticker': str, 'Feat_Time': str})


def get_common_fs(
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


def calculate_roll_mean(df, window):
    meta = df[['Ticker', 'Feat_Time']]
    roll_mean = (
        df
        .groupby('Ticker')
        .rolling(window)
        .mean()
        .shift(-window + 1)
        .reset_index(drop=True)
    )
    roll_mean = roll_mean.add_suffix(f'_Mean_{window}Q')
    roll_mean = pd.concat([meta, roll_mean], axis=1)
    return roll_mean


def shift_data(df, periods):
    meta = df[['Ticker', 'Feat_Time']]
    shift = (
        df
        .drop('Feat_Time', axis=1)
        .groupby('Ticker')
        .shift(-periods)
        .reset_index(drop=True)
    )
    shift = pd.concat([meta, shift], axis=1)
    return shift


def calculate_momentum(df, window, periods):
    meta = df[['Ticker', 'Feat_Time']]

    roll_mean = calculate_roll_mean(df, window)
    shift = shift_data(roll_mean, periods)

    roll_mean = roll_mean.drop(['Ticker', 'Feat_Time'], axis=1)
    shift = shift.drop(['Ticker', 'Feat_Time'], axis=1)

    momentum = roll_mean.div(shift)
    momentum = momentum.add_suffix(f'_Momen_{periods}Q')
    momentum = pd.concat([meta, momentum], axis=1)
    return momentum


def calculate_roll_mean_momentum(df, window_list=[1, 2, 4], periods_list=[1, 2, 4]):
    meta = df[['Ticker', 'Feat_Time']]
    roll_mean_momentum = []
    for window in window_list:
        for periods in periods_list:
            momentum_window_periods = calculate_momentum(df, window, periods)
            momentum_window_periods = momentum_window_periods.drop(['Ticker', 'Feat_Time'], axis=1)
            roll_mean_momentum.append(momentum_window_periods)
    roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
    return roll_mean_momentum


income_momen = calculate_roll_mean_momentum(income)
balance_momen = calculate_roll_mean_momentum(balance)
income_common = get_common_fs(income)
balance_common = get_common_fs(balance)
