from common.genetics import GA
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
from tasks.reg.selectors import ManualSelect, GASelect
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
import seaborn as sns


class GASample(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return GASelect(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        fname = 'sampled_{0}.csv'.format(self.task)
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

            # remove null columns
            useless_cols = [x for x in train.columns if pd.isnull(train[x]).all()]
            train.drop(useless_cols, axis=1, inplace=True)
            valid.drop(useless_cols, axis=1, inplace=True)

            ga = GA(train.values, valid.values, SVR(), iter=10,
                    r_sample=0.5, verbose=True, r_keep_best=0.01)
            (sample, gene, varies) = ga.select_instance()

            #sns.tsplot(varies)
            #plt.show()
            #assert None

            final = pd.concat([final, train_ori[gene], tests_ori])

        #print(final)
        #assert None
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
