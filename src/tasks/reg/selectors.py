import luigi
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pathlib import Path
from common import utils
from sklearn import decomposition
from sklearn import pipeline
from sklearn import feature_selection
from tasks.reg.preprocessor import Preprocess
from common.genetics import GA
from sklearn.svm import SVR


class GASelect(luigi.Task):
    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return Preprocess(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        fname = 'selected_ga_' + Path(self.input().path).name
        outfile = self.dir / fname
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        pool = utils.load_data(self.input().path)

        final = pd.DataFrame()
        for g in pool.groupby(self.kcols):
            (keys, df) = g

            train_ori = df[df.time_window_start < pd.datetime(2016, 10, 18)]
            tests_ori = df[df.time_window_start >= pd.datetime(2016, 10, 18)]
            valid_ori = tests_ori[~pd.isnull(tests_ori[self.vcol])]

            train = train_ori.drop([*self.timecols, *self.kcols], axis=1)
            valid = valid_ori.drop([*self.timecols, *self.kcols], axis=1)

            fcols = train.columns.tolist()
            fcols.remove(self.vcol)

            train = train.reindex_axis([*fcols, self.vcol], axis=1)
            valid = valid.reindex_axis([*fcols, self.vcol], axis=1)

            ga = GA(train.values, valid.values, SVR(), iter=5,
                    r_sample=0.5, verbose=True, r_keep_best=0.01)
            (sample, gene, varies) = ga.select_feature()
            print(train.columns[gene])
            
            useless_cols = train.columns[~gene]
            train_ori[useless_cols] = np.nan

            final = pd.concat([final, train_ori, tests_ori])

        final.to_csv(self.output().path, index=False)

    def _source(self):
        if self.task == 'trajectories':
            vcol = 'avg_travel_time'
            kcols = ['intersection_id', 'tollgate_id']
            on_cols = [*kcols, 'time_window']
        else:
            vcol = 'volume'
            kcols = ['tollgate_id', 'direction']
            on_cols = ['tollgate_id', 'time_window', 'direction']

        self.timecols = ['time_window_start', 'time_window_end']
        self.kcols = kcols
        self.vcol = vcol
        self.oncols = on_cols


class ManualSelect(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()
    select = luigi.BoolParameter(default=False)

    def requires(self):
        return Preprocess(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        fname = 'selected_{0}_'.format(self.select) + Path(self.input().path).name
        outfile = self.dir / fname
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        features = utils.load_data(self.input().path)
        if not self.select:
            features.to_csv(self.output().path, index=False)
            return None

        # drop useless cols via the SVR(traj) or RF(vol)
        features.drop(self.dropcols, axis=1, inplace=True)
        print(features.columns)
        features.to_csv(self.output().path, index=False)

    def _source(self):
        if self.task == 'volume':
            '''
            ['time_window_start', 'time_window_end', 'tollgate_id', 'direction',
            'volume', 'rel_humidity', 'pressure', 'before_holiday',
            'minutes_since_0', 'holiday_len', 'after_holiday', 'precipitation',
            'sea_pressure', 'start_holiday', 'temperature', 'minutes_diff_13',
            'wind_direction', 'neigh_2h_volume_avg', 'neigh_2h_volume_std', 'is_am',
            'wind_speed', 'end_holiday', 'gbdt_0', 'gbdt_1', 'gbdt_2', 'gbdt_3',
            'gbdt_4', 'tsv'],
            '''
            dropcols = [
                'neigh_2h_volume_std',
            ]
        else:
            dropcols = ['neigh_2h_vehicles_std', 'neigh_2h_travel_time_std',
                        'neigh_2h_extracost1_std', 'neigh_2h_extracost1_avg',
                        'rel_humidity', 'wind_speed',
                        'wind_direction', 'sea_pressure', 'pressure',
                        'start_holiday', 'end_holiday',
                        'holiday_len', 'after_holiday']

        self.dropcols = dropcols

