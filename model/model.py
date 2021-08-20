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
        time delta between two time points. Ex: datetime.timedelta(0, 9, 494935)

    Returns
    -------
    three integer objects corresponding to number of hours, minutes and seconds
    '''
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


class Label:

    @staticmethod
    def convert_continuous_to_binary(input, label_col='Return', threshold=0.1):
        output = input.copy()
        output.loc[lambda df: df[label_col] >= threshold, label_col] = 1
        output.loc[lambda df: df[label_col] < threshold, label_col] = 0
        output = output.astype({label_col: int})
        return output

    @staticmethod
    def limit_label_to_first_third_quantiles(input, label_col='Return'):
        first_quartile = np.quantile(input['Return'], 0.25)
        third_quartile = np.quantile(input['Return'], 0.75)
        iqr = third_quartile - first_quartile
        lower_threshold = first_quartile - 1.5 * iqr
        upper_threshold = third_quartile + 1.5 * iqr

        output = input.copy()
        output.loc[output[label_col] < lower_threshold, label_col] = lower_threshold
        output.loc[output[label_col] > upper_threshold, label_col] = upper_threshold
        return output


class SplitData:

    @staticmethod
    def get_train_val_test(
        data,
        label_time_val=None,
        label_time_test=None,
        test_val_period=1,
        label_time_col='Label_Time'
    ):
        if (label_time_val is None) & (label_time_test is None):
            label_time = data[label_time_col].unique()
            label_time = -np.sort(-label_time)

            label_time_test = label_time[:test_val_period]
            label_time_val = label_time[test_val_period: 2 * test_val_period]

        test = data[data[label_time_col].isin(list(label_time_test))]
        val = data[data[label_time_col].isin(list(label_time_val))]
        train = data[~data[label_time_col].isin(list(label_time_test) + list(label_time_val))]
        return train, val, test

    @staticmethod
    def get_label_info(
        train, val, test,
        group_by_time=False,
        label_col='Return',
        label_time_col='Label_Time',
    ):
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

    @staticmethod
    def plot_label(train, val, test):
        data = (
            Label.get_label_info(train, val, test, True)
            .sort_index(ascending=True, level='Label_Time')
            .reset_index()
            .reset_index()
        )
        plt.figure(figsize=(15, 5))
        sns.relplot(x='index', y='mean', kind='line', data=data)
        plt.xticks(data['index'], data['Label_Time'], rotation='vertical', fontsize=10)
        plt.xlabel('')

    @staticmethod
    def split_feat_label_train_val_test(
        train, val, test,
        meta_cols=['Ticker', 'Label_Time', 'Feat_Time'],
        label_col='Return'
    ):
        feat_cols = [col for col in train.columns if col not in meta_cols + [label_col]]

        X_train = train[feat_cols]
        X_val = val[feat_cols]
        X_test = test[feat_cols]

        y_train = train[label_col]
        y_val = val[label_col]
        y_test = test[label_col]
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def convert_to_xgb_data(X_train, y_train, X_val, y_val, X_test, y_test):
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        return dtrain, dval, dtest


class Model:

    @staticmethod
    def convert_params_to_list_dict(params_dict):
        keys, values = zip(*params_dict.items())
        params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return params_list

    @staticmethod
    def grid_search_xgb(
        dtrain,
        dval,
        params_dict,
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=False
    ):
        params_list = Model.convert_params_to_list_dict(params_dict)
        models = []
        results = []
        print(f'There are {len(params_list)} hyperparameter sets.')
        start = datetime.datetime.now()

        for i, params in enumerate(params_list):
            evals_result = {}
            model = xgb.train(
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

            metric_val_max = max(metric_val)
            metric_val_max_index = metric_val.index(metric_val_max)
            gap_at_val_max = metric_gap[metric_val_max_index]
            overfit_max_metric = [metric_val_max, metric_val_max_index, gap_at_val_max]

            # append loop result
            models.append(model)
            results.append(list(params.values()) + overfit_max_metric)

            # print log
            end_i = datetime.datetime.now()
            until_i = end_i - start
            est_total = (end_i - start) * len(params_list) / (i + 1)
            est_remain = est_total - until_i
            hours, minutes, seconds = get_hours_minutes_seconds(est_remain)
            print(f'Finishing {i+1:4}/{len(params_list)} \
                ---> Remaining {hours:02}:{minutes:02}:{seconds:02}')

        results = pd.DataFrame(
            results,
            columns=list(params.keys()) + ['metric_val_max', 'metric_val_max_index', 'gap_at_val_max']
        )
        results = results.drop(['objective', 'eval_metric'], axis=1)
        print('Done')
        return models, results

    @staticmethod
    def get_best_model(models, results, criteria='metric_val_max', higher_better=True):
        best_model_index = results.sort_values(criteria, ascending=1 - higher_better).head(1).index[0]
        best_model = models[best_model_index]
        return best_model


class Evaluation:

    @staticmethod
    def get_calibration(y_pred, y_true, n_group=10):
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

    @staticmethod
    def scatter_plot_prediction(y_pred, y_true, figsize=(5, 5)):
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

    @staticmethod
    def get_feat_imp(model):
        feat_imp = pd.Series(model.get_score(importance_type='gain'))
        feat_imp = feat_imp.sort_values(ascending=False)
        return feat_imp
