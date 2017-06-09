import luigi
from common import utils
from tasks.aggs import Aggregator
from tasks.flats import MergeData
from pathlib import Path
import pandas as pd
import numpy as np
import settings
from datetime import datetime, timedelta
from tasks.reg.features import BasicFeatures


class RollingFeatures(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return BasicFeatures(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        outfile = self.dir / 'feature_rolling_{0}.csv'.format(self.task)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        rawdata = utils.load_data(self.input().path)

        roll_features = self._rolling_time(self.roll_cols, rawdata)
        #roll_features.to_csv('tmp.csv', index=False)
        pool = pd.merge(rawdata, roll_features, how='left', on=self.roll_ons)

        # after generate rolling features, some NA exists
        # we need use the history value to fill NA
        to_fill_cols = set(roll_features.columns) - set([
            'time_window_start', self.key1, self.key2])
        pool = self._fill_na_with_previous(pool, list(to_fill_cols))

        pool.drop(self.extra_cols, axis=1, inplace=True)

        print(pool.shape)
        non_na_cols = set(pool.columns) - set({self.vcol})
        pool.dropna(axis=0, how='any', subset=non_na_cols, inplace=True)
        print(pool.shape)

        pool.to_csv(self.output().path, index=False)

    def _source(self):
        if self.task == 'volume':
            vcol = 'volume'
            key1, key2 = 'tollgate_id', 'direction'
            extra_cols = []
            meta_test_dates = utils.get_meta('volume_tests_dates')
        else:
            vcol = 'avg_travel_time'
            key1, key2 = 'intersection_id', 'tollgate_id'
            extra_cols = ['travel_time', 'vehicles', 'extracost1', 'extracost2']
            meta_test_dates = utils.get_meta('traj_tests_dates')

        self.vcol = vcol
        self.key1, self.key2 = key1, key2
        self.roll_cols = [vcol, *extra_cols]
        self.roll_ons = [key1, key2, 'time_window_start']
        self.extra_cols = extra_cols
        self.meta_test_dates = meta_test_dates

    def _rolling_time(self, cols, pool, hour=2):

        prefix, suffix_avg, suffix_std = 'neigh_{0}h_'.format(hour), '_avg', '_std'

        values = []
        for g in pool.groupby([self.key1, self.key2]):
            ((key1, key2), data) = g

            tmp = data.sort_values('time_window_start', axis=0)
            for itr in tmp.iterrows():
                (index, row) = itr

                starttime = row.time_window_start
                endtime = starttime + timedelta(hours=hour)

                windowdata = tmp[(tmp.time_window_start >= starttime) &
                                 (tmp.time_window_start < endtime)]

                tmpdict = {
                    self.key1: key1, self.key2: key2,
                    # this value if for merge key
                    'time_window_start': endtime + timedelta(hours=2)
                }

                if windowdata.empty:
                    continue

                na_found = False
                for col in cols:
                    if pd.isnull(windowdata[col].mean()):
                        na_found = True

                    tmpdict[prefix + col + suffix_avg] = windowdata[col].mean()
                    tmpdict[prefix + col + suffix_std] = np.std(windowdata[col])
                    # pd.df.std() will return NAN if only one element!

                if na_found:
                    continue

                values.append(tmpdict)

        return pd.DataFrame(values)

    # cols' value will be filled use previous day's value
    def _fill_na_with_previous(self, pool, cols):

        #pool.to_csv('tmp.csv', index=False)

        for col in cols:
            for itr in pool[pool[col].isnull()].iterrows():
                (index, row) = itr

                matchkey1 = pool[self.key1] == row[self.key1]
                matchkey2 = pool[self.key2] == row[self.key2]
                spool = pool[matchkey1 & matchkey2]

                curtime = row.time_window_start
                for i in range(1, 10):
                    targettime = curtime - timedelta(days=i)
                    targetdata = spool[spool.time_window_start == targettime]

                    if targetdata.shape[0] > 1:
                        print(targettime, col, row[self.key1], row[self.key2])
                        raise Exception('Impossible')

                    if targetdata.empty or pd.isnull(targetdata[col].iloc[0]):
                        continue

                    pool.set_value(index, col, targetdata[col])
                    break

        return pool
                
