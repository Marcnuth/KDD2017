import pandas as pd
import numpy as np
import settings
from sklearn.svm import SVR
from common import utils
import luigi
from pathlib import Path
from datetime import time
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from tasks.reg.selectors import ManualSelect
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor
from tasks.reg.sample import GASample


class SkAlgorithm(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()
    algorithm = luigi.Parameter()

    def requires(self):
        return GASample(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        validf = self.dir / 'forecast_valids_{0}_{1}.csv'.format(
            self.task, self.algorithm)
        testf = self.dir / 'forecast_tests_{0}_{1}.csv'.format(
            self.task, self.algorithm)
        return [luigi.LocalTarget(validf.absolute().as_posix()),
                luigi.LocalTarget(testf.absolute().as_posix())]

    def run(self):

        (times_cols, kcols, vcol, submit_cols) = self._source()

        pool = utils.load_data(self.input().path)
        predicts = pd.DataFrame([], columns=pool.columns)

        for g in pool.groupby(kcols):
            (keys, df) = g

            test = df[df.time_window_start >= pd.datetime(2016, 10, 18)]
            train = df[df.time_window_start < pd.datetime(2016, 10, 18)]

            useless_cols = [x for x in train.columns if pd.isnull(train[x]).all()]

            train_x = train.drop(
                [*times_cols, *kcols, vcol, *useless_cols], axis=1)
            train_y = train[vcol]

            regor = self._algorithm().fit(train_x, train_y)
            #if self.algorithm.lower() == 'svr':
            #    print(regor.best_params_)
            #print(regor.estimators_.tolist()[0])

            test_x = test.drop(
                [*times_cols, *kcols, vcol, *useless_cols], axis=1)
            test_y = regor.predict(test_x)
            test[vcol] = test_y
            #print(test_x.head())
            #print(test_y)
            #print(test.head())

            predicts = pd.concat([predicts, test])

        valids = self._fetch(predicts, utils.get_meta('valids_times'), kcols)
        tests = self._fetch(predicts, utils.get_meta('tests_times'), kcols)

        valids = utils.merge_time_window(valids[[*times_cols, *kcols, vcol]])
        tests = utils.merge_time_window(tests[[*times_cols, *kcols, vcol]])

        # rearrange columns
        valids = valids[submit_cols]
        tests = tests[submit_cols]

        valids.to_csv(self.output()[0].path, index=False)
        tests.to_csv(self.output()[1].path, index=False)

    def _algorithm(self):
        if self.algorithm.lower() == 'svr':
            tuned_parameters = [{
                'C': np.arange(1, 4, 0.5),
                'epsilon': np.arange(0.5, 2, 0.2),
                'tol': [1, 1e-1, 1e-2, 1e-3]
            }]
            #return GridSearchCV(
            #    SVR(kernel='rbf', shrinking=True, gamma='auto'), tuned_parameters,
            #    cv=5, error_score=0, n_jobs=4, verbose=1
            #)
            return SVR(kernel='rbf', C=1)
        elif self.algorithm.lower() == 'mlp':
            return MLPRegressor()
        elif self.algorithm.lower() == 'huber':
            return HuberRegressor()
        elif self.algorithm.lower() == 'lr':
            return linear_model.LinearRegression()
        elif self.algorithm.lower() == 'rigid':
            return linear_model.Ridge(alpha=0.5)
        elif self.algorithm.lower() == 'rf':
            return RandomForestRegressor(random_state=0, n_estimators=200)
        elif self.algorithm.lower() == 'gbr':
            tuned_parameters = [{
                'n_estimators': [160, 170, 180],
                'subsample': [0.6, 0.7, 0.8],
            }]
            '''
            return GridSearchCV(
                GradientBoostingRegressor(
                    loss='ls', warm_start=False, max_features=0.2,
                    learning_rate=0.05, alpha=0.4, max_depth=13, subsample=0.6,
                    n_estimators=180),
                tuned_parameters,
                cv=5, error_score=0, n_jobs=4, verbose=1)
            '''
            return GradientBoostingRegressor(
                loss='ls', warm_start=False, max_features=0.2,
                learning_rate=0.05, alpha=0.4, max_depth=13, subsample=0.6,
                n_estimators=180)
        elif self.algorithm.lower() == 'adb':
            return AdaBoostRegressor()
        else:
            raise Exception('Sklearn Algorithm Options: svr')

    def _fetch(self, predicts, metas, kcols):

        predicts['in_meta'] = predicts.time_window_start.map(
            lambda x: [x.time().hour, x.time().minute] in metas
        )

        final = predicts[predicts['in_meta']]

        final.tollgate_id = final.tollgate_id.map(int)
        if self.task == 'volume':
            final.direction = final.direction.map(int)

        final.sort_values(by=[*kcols, 'time_window_start'], axis=0, inplace=True)

        return final

    def _source(self):
        times_cols = ['time_window_start', 'time_window_end']
        if self.task == 'trajectories':
            vcol = 'avg_travel_time'
            kcols = ['intersection_id', 'tollgate_id']
            submit_cols = [*kcols, 'time_window', vcol]
        else:
            vcol = 'volume'
            kcols = ['tollgate_id', 'direction']
            submit_cols = ['tollgate_id', 'time_window', 'direction', vcol]

        return(times_cols, kcols, vcol, submit_cols)


class Voter(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return [SkAlgorithm(self.uuid, self.task, 'svr'),
                SkAlgorithm(self.uuid, self.task, 'gbr')]

    def output(self):
        self.dir = Path(self.input()[0][0].path).absolute().parent
        validf = self.dir / 'forecast_valids_{0}_voter.csv'.format(
            self.task)
        testf = self.dir / 'forecast_tests_{0}_voter.csv'.format(
            self.task)
        return [luigi.LocalTarget(validf.absolute().as_posix()),
                luigi.LocalTarget(testf.absolute().as_posix())]

    def run(self):
        (kcols, vcol, on_cols) = self._source()

        svr_valids = utils.load_data(self.input()[0][0].path)
        svr_tests = utils.load_data(self.input()[0][1].path)
        gbr_valids = utils.load_data(self.input()[1][0].path)
        gbr_tests = utils.load_data(self.input()[1][1].path)

        valids = pd.merge(svr_valids, gbr_valids, how='left', on=on_cols)
        tests = pd.merge(svr_tests, gbr_tests, how='left', on=on_cols)

        valids[vcol] = (valids[vcol + '_x'] * 2 + valids[vcol + '_y'] * 1) / 3.0
        tests[vcol] = (tests[vcol + '_x'] * 2 + tests[vcol + '_y'] * 1) / 3.0

        to_drops = [vcol + '_x', vcol + '_y']
        valids.drop(to_drops, axis=1, inplace=True)
        tests.drop(to_drops, axis=1, inplace=True)

        valids.to_csv(self.output()[0].path, index=False)
        tests.to_csv(self.output()[1].path, index=False)

    def _source(self):
        if self.task == 'trajectories':
            vcol = 'avg_travel_time'
            kcols = ['intersection_id', 'tollgate_id']
            on_cols = [*kcols, 'time_window']
        else:
            vcol = 'volume'
            kcols = ['tollgate_id', 'direction']
            on_cols = ['tollgate_id', 'time_window', 'direction']

        return (kcols, vcol, on_cols)
