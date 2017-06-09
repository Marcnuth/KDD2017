import luigi
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from common import utils
from datetime import timedelta
from tasks.aggs import Aggregator
from tasks.tsa.extractor import ExtractTS

logger = logging.getLogger('luigi-interface')
logger.setLevel('INFO')


class FillNA(luigi.Task):
    uuid = luigi.Parameter()
    resource = luigi.Parameter()
    time_range = luigi.Parameter()

    def requires(self):
        return [Aggregator(self.uuid, self.resource),
                ExtractTS(self.uuid, self.resource, self.time_range)]

    def output(self):
        self.dir = Path(self.input()[0].path).absolute().parent
        path = self.input()[1].path
        outfile = path[:path.rindex('.')] + '_no_nan.csv'
        return luigi.LocalTarget(outfile)

    def run(self):

        pool = utils.load_data(self.input()[0].path)
        tss = utils.load_data(self.input()[1].path)

        final = self._fill_na(pool, tss)
        final.to_csv(self.output().path, index=False)

    def _source(self):
        is_volume = 'volume' in self.resource
        if is_volume:
            return ('volume', self._smaller_vol_pool,)
        else:
            return ('avg_travel_time', self._smaller_traj_pool)

    def _smaller_traj_pool(self, pool, row):
        iid = row.intersection_id
        tid = row.tollgate_id

        return pool[(pool.intersection_id == iid) & (pool.tollgate_id == tid)]

    def _smaller_vol_pool(self, pool, row):
        tid = row.tollgate_id
        direct = row.direction

        return pool[(pool.tollgate_id == tid) & (pool.direction == direct)]

    def _fill_na(self, pool, data):
        (col, poolfunc) = self._source()

        nullrows = data[data[col].isnull()]
        for row in nullrows.iterrows():

            tmppool = poolfunc(pool, row[1])

            esti = self._avg_round_time(tmppool, row[1].time_window_start, col)
            if pd.isnull(esti):
                esti = self._avg_week_time(tmppool, row[1].time_window_start, col)

            if pd.isnull(esti):
                print('======> Use avg')
                esti = tmppool[col].median()

            assert not pd.isnull(esti), 'Not find any proper value for NAN!'
            data.set_value(row[0], col, esti)

        return data

    def _avg_round_time(self, pool, time_window_start, col):
        prev_dt = time_window_start - timedelta(hours=1)
        next_dt = time_window_start + timedelta(hours=1)

        filtered = pool[(pool.time_window_start <= next_dt) &
                        (pool.time_window_start >= prev_dt)]

        #print(filtered)
        #print(filtered[col].mean())

        return filtered[col].mean()

    def _avg_week_time(self, pool, time_window_start, col):
        prev_week = time_window_start - timedelta(days=7)
        next_week = time_window_start + timedelta(days=7)

        prev_avg = self._avg_round_time(pool, prev_week, col)
        next_avg = self._avg_round_time(pool, next_week, col)

        if pd.isnull(prev_avg) and pd.isnull(next_avg):
            return np.nan
        elif not pd.isnull(prev_avg) and pd.isnull(next_avg):
            return prev_avg
        elif pd.isnull(prev_avg) and not pd.isnull(next_avg):
            return next_avg
        else:
            return np.average([prev_avg, next_avg])

