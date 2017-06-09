import luigi
from tasks.aggs import Aggregator
from pathlib import Path
from common import utils
from datetime import time, timedelta
import pandas as pd


class ExtractTS(luigi.Task):

    uuid = luigi.Parameter()
    resource = luigi.Parameter()
    time_range = luigi.Parameter()

    def requires(self):
        return Aggregator(self.uuid, self.resource)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        self.metas = self._source()[0]
        outfile = self.dir / 'ts_{0}__{1}_{2}.csv'.format(self.uuid, self.resource, self.time_range)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):

        pool = utils.load_data(self.input().path)
        filter_ts = self._source()[1]

        final = pd.DataFrame([], columns=pool.columns, dtype=pool.dtypes)
        for i in self.metas:
            #logger.info(i)
            data = filter_ts(pool, *i)
            final = pd.concat([final, data])

        final.to_csv(self.output().path, index=False)

    def _source(self):
        is_valids = 'valids' == self.time_range
        is_volume = 'volume' in self.resource

        if is_valids and is_volume:
            return (utils.get_meta('volume_valids_times'), self._filter_vol_ts)
        elif is_valids and not is_volume:
            return (utils.get_meta('traj_valids_times'), self._filter_traj_ts)
        elif not is_valids and is_volume:
            return (utils.get_meta('volume_tests_times'), self._filter_vol_ts)
        else:
            return (utils.get_meta('traj_tests_times'), self._filter_traj_ts)

    def _filter_traj_ts(self, pool, iid, tid, hour, minute):
        tmp = pool[(pool.intersection_id == iid) & (pool.tollgate_id == tid)]
        tmp.index = tmp.time_window_start
        data = tmp.select(lambda x: x.time() == time(hour, minute), axis=0)

        data.index = data.time_window_start
        data.sort_index(inplace=True)

        ideal = pd.DataFrame(index=pd.date_range(data.index[0], data.index[-1]))
        final = pd.concat([data, ideal], axis=1)
        final.intersection_id = iid
        final.tollgate_id = tid
        final.time_window_start = final.index
        final.time_window_end = final.index + timedelta(minutes=20)

        return final

    def _filter_vol_ts(self, pool, tid, direct, hour, minute):
        tmp = pool[(pool.tollgate_id == tid) & (pool.direction == direct)]
        tmp.index = tmp.time_window_start
        data = tmp.select(lambda x: x.time() == time(hour, minute), axis=0)

        data.index = data.time_window_start
        data.sort_index(inplace=True)

        ideal = pd.DataFrame(index=pd.date_range(data.index[0], data.index[-1]))
        final = pd.concat([data, ideal], axis=1)

        # reset the constant values just in case some value missing
        final.tollgate_id = tid
        final.time_window_start = final.index
        final.time_window_end = final.index + timedelta(minutes=20)
        final.direction = direct

        return final
