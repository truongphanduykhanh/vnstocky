'''
This script is to generate model for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-08-19'


import itertools
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


def get_hours_minutes_seconds(timedelta):
    '''
    Convert time delta to hours, minutes, seconds

    Parameters
    ----------
    timedelta : datetime.timedelta
        Time delta between two time points. Ex: datetime.timedelta(0, 9, 494935).

    Returns
    -------
    three integers
        Corresponding to number of hours, minutes and seconds.
    '''
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def convert_continuous_to_binary(input, label_col='Return', threshold=0.1):
    '''
    Convert continuous label to binary

    Parameters
    ----------
    input : pandas.DataFrame
        Fata frame including label column.
    label_col : str
        Label column name.
    threshold : float
        Threshold at which the continuous values change from 0 to 1.

    Returns
    -------
    pandas.DataFrame
        Data frame that includes binary label.
    '''
    output = input.copy()
    output.loc[lambda df: df[label_col] >= threshold, label_col] = 1
    output.loc[lambda df: df[label_col] < threshold, label_col] = 0
    output = output.astype({label_col: int})
    return output


def limit_label_to_first_third_quantiles(input, label_col='Return'):
    '''
    Change continuous label to first quantile for those are less than first quantile,
    and to third quantile for those are greater than third quantile

    Parameters
    ----------
    input : pandas.DataFrame
        Data frame including label column.
    label_col : str
        Label column name.

    Returns
    -------
    pandas.DataFrame
        Data frame that includes binary label.
    '''
    first_quartile = np.quantile(input['Return'], 0.25)
    third_quartile = np.quantile(input['Return'], 0.75)
    iqr = third_quartile - first_quartile
    lower_threshold = first_quartile - 1.5 * iqr
    upper_threshold = third_quartile + 1.5 * iqr

    output = input.copy()
    output.loc[output[label_col] < lower_threshold, label_col] = lower_threshold
    output.loc[output[label_col] > upper_threshold, label_col] = upper_threshold
    return output


def get_train_val_test(
    data,
    test_val_period=1,
    label_time_val=None,
    label_time_test=None,
    label_time_col='Label_Time'
):
    '''
    Split data to train, validation and test sets

    Parameters
    ----------
    data : pandas.DataFrame
        Data frame including label time column.
    test_val_period : int
        Number of periods for test and validation set.
        Will be ignored if label_time_val and label_time_test are set.
    label_time_val : str
        Period in string for validation set.
    label_time_test : str
        Period in string for test set.
    label_time_col : str
        Label time column name.

    Returns
    -------
    three pandas.DataFrame
        Train, validation and test sets.
    '''
    if (label_time_val is None) & (label_time_test is None):
        label_time = data[label_time_col].unique()
        label_time = -np.sort(-label_time)

        label_time_test = label_time[:test_val_period]
        label_time_val = label_time[test_val_period: 2 * test_val_period]

    test = data[data[label_time_col].isin(list(label_time_test))]
    val = data[data[label_time_col].isin(list(label_time_val))]
    train = data[~data[label_time_col].isin(list(label_time_test) + list(label_time_val))]
    return train, val, test


def get_label_info(
    train, val, test,
    group_by_time=False,
    label_col='Return',
    label_time_col='Label_Time',
):
    '''
    Get summary information of label in train, validation and test sets

    Parameters
    ----------
    train, val, test : pandas.DataFrame
        Train, validation and test data sets.
    group_by_time : bool
        If get summary according to time.
    label_col : str
        Label column name.
    label_time_col : str
        Label time column name.

    Returns
    -------
    pandas.DataFrame
        Summary of label.
    '''
    data = (
        pd.concat([
            train.assign(Set='train'),
            val.assign(Set='val'),
            test.assign(Set='test')], axis=0))
    if group_by_time:
        data = (
            data
            .groupby(['Set', label_time_col])
            .agg({label_col: ['count', 'sum', 'mean']})
            .sort_index(ascending=False, level='Label_Time')
            .set_axis(['count', 'sum', 'mean'], axis=1)
        )
    else:
        data = (
            data
            .groupby('Set')
            .agg({label_col: ['count', 'sum', 'mean']})
            .loc[['train', 'val', 'test'], :]
            .set_axis(['count', 'sum', 'mean'], axis=1)
        )
    return data


def plot_label(train, val, test, label_time_col='Label_Time'):
    '''
    Parameters
    ----------
    train, val, test : pandas.DataFrame
        Train, validation and test data sets.
    label_time_col : str
        Label time column name.

    Returns
    -------
    plot
        Plot of label through out the time.
    '''
    data = (
        get_label_info(train, val, test, True)
        .sort_index(ascending=True, level=label_time_col)
        .reset_index()
        .reset_index()
    )
    plt.figure(figsize=(15, 5))
    sns.relplot(x='index', y='mean', kind='line', data=data)
    plt.xticks(data['index'], data[label_time_col], rotation='vertical', fontsize=10)
    plt.xlabel('')


def split_feat_label_train_val_test(
    train, val, test,
    meta_cols=['Ticker', 'Label_Time', 'Feat_Time'],
    label_col='Return'
):
    '''
    Get summary information of label in train, validation and test sets

    Parameters
    ----------
    train, val, test : pandas.DataFrame
        Train, validation and test data sets.
    meta_cols : list of strings
        List of names of meta columns.
    label_col : str
        Label column name.

    Returns
    -------
    six pandas.DataFrame and pandas.Series
        X_train, y_train, X_val, y_val, X_test, y_test
    '''
    feat_cols = [col for col in train.columns if col not in meta_cols + [label_col]]

    X_train = train[feat_cols]
    X_val = val[feat_cols]
    X_test = test[feat_cols]

    y_train = train[label_col]
    y_val = val[label_col]
    y_test = test[label_col]
    return X_train, y_train, X_val, y_val, X_test, y_test


def convert_to_xgb_data(X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    Convert pandas data frame and series to XGBoost type

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test : pandas.DataFrame and pandas.Series
        Train, validation and test data sets.

    Returns
    -------
    three data XGBoost type
        dtrain, dval, dtest
    '''
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    return dtrain, dval, dtest


def convert_params_to_list_dict(params_dict):
    '''
    Convert dictionary to list of dictionaries

    Parameters
    ----------
    params_dict : dict
        Dictionary of parameters.

    Returns
    -------
    list of dict
        List of dictionaries, ready to grid search.
    '''
    keys, values = zip(*params_dict.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return params_list


def grid_search_xgb(
    dtrain,
    dval,
    params_dict,
    num_boost_round=1000,
    early_stopping_rounds=100,
    verbose_eval=False
):
    '''
    Perform grid search through many sets of hyperparameters

    Parameters
    ----------
    dtrain : XGBoost data
        Training data.
    dval : XGBoost data
        Validation data.
    params_dict : dict
        Dictionary of hyperparameters.
    num_boost_round : int
        Number of boosting iterations.
    early_stopping_rounds : int
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
    verbose_eval : bool or int
        Print log during training.

    Returns
    -------
    pandas.DataFrame
        results of grid search
    '''
    params_list = convert_params_to_list_dict(params_dict)
    print(f'There are {len(params_list)} hyperparameter sets.')
    global grid_search
    grid_search = []
    start = datetime.datetime.now()

    for i, params in enumerate(params_list):
        evals_result = {}
        booster = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=[(dtrain, 'dtrain'), (dval, 'dval')],
            verbose_eval=verbose_eval,
            evals_result=evals_result
        )

        metric = list(evals_result['dtrain'].keys())[-1]
        metric_train = evals_result['dtrain'][metric]
        metric_val = evals_result['dval'][metric]
        metric_gap = [x - y for x, y in zip(metric_train, metric_val)]

        metric_val_last = metric_val[-1]
        metric_val_max = max(metric_val)
        metric_val_max_index = metric_val.index(metric_val_max)
        gap_at_val_max = metric_gap[metric_val_max_index]
        overfit_max_metric = [
            metric_val_last, metric_val_max, metric_val_max_index, gap_at_val_max]
        grid_search_i = list(params.values()) + overfit_max_metric

        # append loop result
        grid_search.append(grid_search_i)

        # print log
        end_i = datetime.datetime.now()
        until_i = end_i - start
        est_total = (end_i - start) * len(params_list) / (i + 1)
        est_remain = est_total - until_i
        hours, minutes, seconds = get_hours_minutes_seconds(est_remain)
        print(f'Finishing {i+1:4}/{len(params_list)} \
            ---> Remaining {hours:02}:{minutes:02}:{seconds:02}')

    grid_search = pd.DataFrame(
        grid_search,
        columns=list(params.keys()) + \
        ['metric_val_last', 'metric_val_max', 'metric_val_max_index', 'gap_at_val_max']
    )
    grid_search = grid_search.drop(['objective', 'eval_metric'], axis=1)
    print('Done')
    return grid_search


def get_best_model(dtrain, params_dict, grid_search, criteria='metric_val_max', higher_better=True):
    '''
    Select the best model from grid search

    Parameters
    ----------
    dtrain : XGBoost data
        Training data.
    params_dict : dict
        Dictionary of hyperparameters.
    grid_search : pandas.DataFrame
        Results of grid search.
    criteria : str
        Column name of metric in results on that is based to select best model.
    higher_better : bool
        If the metric is the higher the better.

    Returns
    -------
    Booster
        Best XGBoost model.
    '''
    best_model_index = grid_search.sort_values(criteria, ascending=1 - higher_better).head(1).index[0]
    best_model_params = convert_params_to_list_dict(params_dict)[best_model_index]

    booster = xgb.train(
        best_model_params,
        dtrain=dtrain,
        num_boost_round=grid_search.loc[best_model_index]['metric_val_max_index'] + 1,
        verbose_eval=False
    )
    return booster


def get_calibration(y_pred, y_true, n_group=10):
    '''
    Get calibration data frame

    Parameters
    ----------
    y_pred : list of float
        Prediction from model.
    y_true : list of float
        Actual values.
    n_group : int
        Number of group in the calibration.

    Returns
    -------
    pandas.DataFrame
        Calibration data frame.
    '''
    calibration = pd.DataFrame({'score': y_pred, 'true': y_true})
    calibration = (
        calibration
        .assign(group=lambda df: n_group - pd.qcut(df['score'], n_group, labels=False))
        .groupby('group')
        .agg({
            'score': np.mean,
            'true': [np.mean, 'count', np.sum]})
        .set_axis(['mean_score', 'mean_true', 'count_true', 'sum_true'], axis=1)
        .astype({'sum_true': int})
    )
    return calibration


def scatter_plot_prediction(y_pred, y_true, figsize=(5, 5)):
    '''
    Get scatter plot of prediction

    Parameters
    ----------
    y_pred : list of float
        Prediction from model.
    y_true : list of float
        Actual values.
    figsize : tuble
        Size of the plot.

    Returns
    -------
    seaborn plot
        Scatter plof of prediction.
    '''
    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_pred, y=y_true)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Prediction', size=12)
    plt.ylabel('Actual', size=12)
    plt.title('Prediction', size=15)
    plt.axline([0, 0], [1, 1], color='red', ls='-.')
    for x in [-0.1, 0.1]:
        plt.axvline(x=x, color='green', linestyle=':')
    for y in [-0.1, 0.1]:
        plt.axhline(y=y, color='green', linestyle=':')


def get_feat_imp(model):
    '''
    Get feature imporatance of model

    Parameters
    ----------
    model : Booster
        XGBoost model.

    Returns
    -------
    pandas.Series
        Feature importance, from higest for lowest.
    '''
    feat_imp = pd.Series(model.get_score(importance_type='gain'))
    feat_imp = feat_imp.sort_values(ascending=False)
    return feat_imp
