import luigi
from common import utils
from tasks.aggs import Aggregator
from tasks.flats import MergeData
from pathlib import Path
import pandas as pd
import numpy as np
import settings
from datetime import datetime, timedelta
from tasks.tsa.model import ARIMAForecast


class BasicFeatures(luigi.Task):
    '''
    Basic features include:
    original: ids/timewindows/value ...
    minutes_since_0: 9:20 = 9 * 60 + 20
    minutes_diff_13: 9:20 am = abs(9-13) * 60 + 20,  18: 40 pm = (18-13) * 60 + 20
    before_holiday: 1 or 0: 1 means true, 0 for false (the before's diff is day=1)
    after_holiday: 1 or 0, the after's diff is day=1
    holiday_len: 0 for not holiday, 2 for weekend, 7 for national day, etc.
    is_am:  before 12:00 am = 1, after = 0
    weather: feaatures from weather.csv
    '''

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        if self.task == 'volume':
            return [Aggregator(self.uuid, 'train_volume'),
                    Aggregator(self.uuid, 'valids_volume')]
        else:
            return [MergeData(self.uuid, 'train_trajectories'),
                    MergeData(self.uuid, 'valids_trajectories')]

    def output(self):
        self.dir = Path(self.input()[0].path).absolute().parent
        outfile = self.dir / 'feature_basic_{0}.csv'.format(self.task)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        pool = self._ts_features()
        weathers = self._weathers()

        final = self._merge(pool, weathers)
        # do not remove, so we can revert those after predict
        #final.drop(['time_window_start', 'time_window_end'], axis=1, inplace=True)
        final.to_csv(self.output().path, index=False)

    def _tests_metas(self):
        time_cols = ['time_window_start', 'time_window_end']
        cols = [self.key1, self.key2, *time_cols, self.vcol, *self.extra_cols]

        datas = []
        for item in self.meta_test_dates:
            (year, mon, day, k1, k2, hour, minute) = item
            #print(item)
            btime = datetime(year, mon, day, hour, minute)
            datas.append([k1, k2, btime, btime + timedelta(minutes=20), np.nan,
                          *[np.nan for i in range(len(self.extra_cols))]])

        return pd.DataFrame(datas, columns=cols)

    def _ts_features(self):

        pool1 = utils.load_data(self.input()[0].path)  # train
        pool2 = utils.load_data(self.input()[1].path)  # valids
        pool3 = self._tests_metas()                    # tests
        pool = pd.concat([pool1, pool2, pool3])
        pool.reset_index(drop=True, inplace=True)

        pool['minutes_since_0'] = pool.time_window_start.map(
            lambda x: x.hour * 60 + x.minute
        )
        pool['minutes_diff_13'] = pool.time_window_start.map(
            lambda x: abs(x.hour - 13) * 60 + x.minute
        )
        pool['before_holiday'] = pool.time_window_start.map(utils.before_holiday)
        pool['after_holiday'] = pool.time_window_start.map(utils.after_holiday)
        pool['start_holiday'] = pool.time_window_start.map(utils.start_holiday)
        pool['end_holiday'] = pool.time_window_start.map(utils.end_holiday)
        pool['holiday_len'] = pool.time_window_start.map(utils.holiday_len)
        pool['is_am'] = pool.time_window_start.map(lambda x: x.hour > 13)

        return pool

    def _weathers(self):

        wea1 = pd.read_csv(settings.Data.Train.weather)
        wea2 = pd.read_csv(settings.Data.Test.weather)
        weather = pd.concat([wea1, wea2])

        weather.date = pd.to_datetime(weather.date)
        weather.hour = pd.to_numeric(weather.hour)
        weather.sort_values(by=['date', 'hour'], axis=0, inplace=True)

        # weather lacking data, we need to fill it using rounding values
        tmp = []
        for dt in pd.date_range(weather.date.iloc[0], weather.date.iloc[-1]):
            for h in range(0, 24, 3):
                tmp.append([dt, h])

        ideal = pd.DataFrame(tmp, columns=('date', 'hour'))
        final = pd.merge(ideal, weather, how='left', on=['date', 'hour'])

        final.fillna(method='ffill', inplace=True)
        return final

    def _merge(self, pool, weather):

        # tmp column for merging weather
        pool['date'] = pool.time_window_start.map(
            lambda x: x.strftime('%Y-%m-%d')
        )
        pool['hour'] = pool.time_window_start.map(
            lambda x: int(np.ceil(x.hour / 3) * 3) % 24
        )

        # just in case
        weather.date = pd.to_datetime(weather.date)
        weather.hour = pd.to_numeric(weather.hour)
        pool.date = pd.to_datetime(pool.date)
        pool.hour = pd.to_numeric(pool.hour)

        pool = pd.merge(pool, weather, how='left', on=['date', 'hour'])
        pool.drop(['date', 'hour'], axis=1, inplace=True)

        return pool

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


class ARIMAFeatures(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        if self.task == 'trajectories':
            return [ARIMAForecast(self.uuid, 'train_trajectories', 'valids'),
                    ARIMAForecast(self.uuid, 'train_trajectories', 'tests')]
        else:
            return [ARIMAForecast(self.uuid, 'train_volume', 'valids'),
                    ARIMAForecast(self.uuid, 'train_volume', 'tests')]

    def output(self):
        self.dir = Path(self.input()[0][0].path).absolute().parent
        outfile = self.dir / 'feature_arima_{0}.csv'.format(self.task)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):

        fitted_valids = utils.load_data(self.input()[0][1].path)
        fitted_tests = utils.load_data(self.input()[1][1].path)

        final = pd.concat([fitted_valids, fitted_tests])
        final.to_csv(self.output().path, index=False)
