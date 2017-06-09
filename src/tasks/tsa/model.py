import luigi
import logging
from pathlib import Path
from tasks.tsa.filler import FillNA
from common import utils
import pandas as pd
import numpy as np
import csv
from datetime import time, datetime, timedelta
from tasks.aggs import Aggregator
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger('luigi-interface')
logger.setLevel('INFO')


class ARIMAForecast(luigi.Task):
    uuid = luigi.Parameter()
    resource = luigi.Parameter()
    time_range = luigi.Parameter()

    def requires(self):
        return FillNA(self.uuid, self.resource, self.time_range)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        outfile = self.dir / 'forecast_{1}_{0}_arima.csv'.format(
            self.resource, self.time_range)
        fitfile = self.dir / 'fitted_{0}'.format(outfile.name)
        return [luigi.LocalTarget(outfile.absolute().as_posix()),
                luigi.LocalTarget(fitfile.absolute().as_posix())]

    def run(self):
        (metas, tsfunc, genfunc, key1, key2, vcol) = self._source()

        dates = utils.get_meta('dates')

        fitted_cols = [key1, key2, 'time_window_start', 'tsv']
        forecasts, fitted = {}, pd.DataFrame([], columns=fitted_cols)

        pool = utils.load_data(self.input().path)
        #logger.info(pool)
        for meta in metas:
            ts = tsfunc(pool, meta)

            to_forecast_dates = pd.datetime(*dates[-1]).date() - ts.index[-1].date()
            (forecast, fit) = utils.fit_arima(ts, to_forecast_dates.days)

            forecasts[str(meta)] = forecast[-1 * len(dates):]
            assert len(forecasts[str(meta)]) == len(dates), 'Code is wrong!'

            # append fitted values
            tmp = pd.DataFrame(fit, columns=['tsv'])
            tmp['time_window_start'] = tmp.index
            tmp[key1] = meta[0]
            tmp[key2] = meta[1]

            fitted = pd.concat([fitted, tmp[fitted_cols]])

        # output the predict csv & fitted csv
        # predict csv should following submit sample(with time_window)
        # fitted csv has no time_window, but has time_window_start
        final = []
        for meta in metas:
            forecast = forecasts[str(meta)]
            for i in range(len(dates)):
                #logger.info(meta)
                #logger.info(dates[i])
                #logger.info(forecast)
                (year, month, day) = dates[i]
                final.append([*meta, year, month, day, forecast[i]])

        finaldf = genfunc(final)
        finaldf.to_csv(self.output()[0].path, index=False)

        # output fitted values, merge the predicts values into here
        finaldf['tsv'] = finaldf[vcol]
        finaldf['time_window_start'] = finaldf.time_window.map(
            lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S')
        )
        finaldf.drop(['time_window', vcol], axis=1, inplace=True)

        fitted = pd.concat([fitted, finaldf])
        fitted.to_csv(self.output()[1].path, index=False)

    def _gen_traj_result(self, datas):

        tmp = []
        for row in datas:
            (iid, tid, hour, minute, year, mon, day, value) = row
            tfmt = '%Y-%m-%d %H:%M:%S'
            bastime = datetime(year, mon, day, hour, minute)
            tstart = bastime.strftime(tfmt)
            tend = (bastime + timedelta(minutes=20)).strftime(tfmt)
            time_window = '[' + tstart + ',' + tend + ')'
            tmp.append([iid, int(tid), time_window, value])

        cols = ('intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time')
        final = pd.DataFrame(tmp, columns=cols)
        final.sort(['intersection_id', 'tollgate_id', 'time_window'], inplace=True)
        return final

    def _gen_vol_result(self, datas):

        tmp = []
        for row in datas:
            (tid, direction, hour, minute, year, mon, day, value) = row
            tfmt = '%Y-%m-%d %H:%M:%S'
            basetime = datetime(year, mon, day, hour, minute)
            tstart = basetime.strftime(tfmt)
            tend = (basetime + timedelta(minutes=20)).strftime(tfmt)
            time_window = '[' + tstart + ',' + tend + ')'
            tmp.append([int(tid), time_window, int(direction), value])

        cols = ('tollgate_id', 'time_window', 'direction', 'volume')
        final = pd.DataFrame(tmp, columns=cols)
        final.sort(['tollgate_id', 'time_window', 'direction'], inplace=True)
        return final

    def _source(self):
        is_valids = 'valids' == self.time_range
        is_volume = 'volume' in self.resource

        if is_valids and is_volume:
            return (utils.get_meta('volume_valids_times'),
                    self._get_vol_ts,
                    self._gen_vol_result,
                    'tollgate_id', 'direction', 'volume')
        elif is_valids and not is_volume:
            return (utils.get_meta('traj_valids_times'),
                    self._get_traj_ts,
                    self._gen_traj_result,
                    'intersection_id', 'tollgate_id', 'avg_travel_time')
        elif not is_valids and is_volume:
            return (utils.get_meta('volume_tests_times'),
                    self._get_vol_ts,
                    self._gen_vol_result,
                    'tollgate_id', 'direction', 'volume')
        else:
            return (utils.get_meta('traj_tests_times'),
                    self._get_traj_ts,
                    self._gen_traj_result,
                    'intersection_id', 'tollgate_id', 'avg_travel_time')

    # meta: a tuple: (iid, tid, hour, minute)
    def _get_traj_ts(self, pool, meta):
        (iid, tid, hour, minute) = meta

        tmp = pool[(pool.intersection_id == iid) & (pool.tollgate_id == tid)]
        tmp.index = tmp.time_window_start
        data = tmp.select(lambda x: x.time() == time(hour, minute), axis=0)

        ts = data.avg_travel_time
        ts.index = data.time_window_start

        return ts

    # meta: (tid, direction, hour, minute)
    def _get_vol_ts(self, pool, meta):
        (tid, direction, hour, minute) = meta

        tmp = pool[(pool.direction == direction) & (pool.tollgate_id == tid)]
        tmp.index = tmp.time_window_start
        data = tmp.select(lambda x: x.time() == time(hour, minute), axis=0)

        ts = data.volume
        ts.index = data.time_window_start

        return ts


class TSARIMA(luigi.Task):

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
        validsf = self.dir / 'forecast_valids_tsarima_{0}.csv'.format(self.task)
        testsf = self.dir / 'forecast_tests_tsarima_{0}.csv'.format(self.task)
        return [luigi.LocalTarget(validsf.absolute().as_posix()),
                luigi.LocalTarget(testsf.absolute().as_posix())]

    def run(self):

        valids = utils.load_data(self.input()[0][0].path)
        valids.to_csv(self.output()[0].path, index=False)

        tests = utils.load_data(self.input()[1][0].path)
        tests.to_csv(self.output()[1].path, index=False)

