import luigi
import numpy as np
from pathlib import Path
from common import utils
from tasks.reg.rollings import RollingFeatures


class PolyFeatures(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return RollingFeatures(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        fname = 'feature_ploy_' + Path(self.input().path).name
        outfile = self.dir / fname
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        df = utils.load_data(self.input().path)

        # remove those data not in valids or tests time
        # those invalids features only used for rollings
        features = self._remove_not_in_times(df)

        # generate some ploy features
        features = self.ploy_func(features)

        features.to_csv(self.output().path, index=False)

    def _remove_not_in_times(self, df):

        final = df[df.time_window_start.map(
            lambda x: x.time().hour in [6, 7, 8, 9, 15, 16, 17, 18]
        )]

        return final

    def _traj_ploy_features(self, df):
        def __np_bool(x):
            return 1 if x else -1

        '''
        allcols = [
            'minutes_since_0', 'minutes_diff_13', 'before_holiday', 'after_holiday',
            'start_holiday', 'end_holiday',
            'holiday_len', 'is_am', 'pressure', 'sea_pressure', 'wind_direction',
            'wind_speed', 'temperature', 'rel_humidity', 'precipitation',
            'neigh_2h_avg_travel_time_avg', 'neigh_2h_avg_travel_time_std',
            'neigh_2h_extracost1_avg', 'neigh_2h_extracost1_std',
            'neigh_2h_extracost2_avg', 'neigh_2h_extracost2_std',
            'neigh_2h_travel_time_avg', 'neigh_2h_travel_time_std',
            'neigh_2h_vehicles_avg', 'neigh_2h_vehicles_std']
        '''

        df['is_am_min_diff_13'] = df['is_am'].map(__np_bool) * df['minutes_diff_13']

        df['surprise1'] = (df['before_holiday'] | df['end_holiday']) & df['is_am']
        df['surprise2'] = df['surprise1'].map(__np_bool) * df['minutes_diff_13']

        #df['surprise3'] = df['after_holiday'].map(__np_bool) * df['minutes_since_0']

        df['surprise3'] = df['minutes_diff_13'] / np.log(
            df['neigh_2h_avg_travel_time_avg'])

        #rains = df['precipitation'] - df['precipitation'].median()
        #workday = df['holiday_len'].map(lambda x: 1 if x > 0 else 0)
        #df['surprise4'] = rains / df['temperature']
        return df

    def _vol_ploy_features(self, df):
        return df

    def _source(self):
        cols = ['time_window_start', 'time_window_end']
        if self.task == 'volume':
            cols.extend(['tollgate_id', 'direction', 'volume'])
            vcol = 'volume'
            ploy_func = self._vol_ploy_features
        else:
            cols.extend(['intersection_id', 'tollgate_id', 'avg_travel_time'])
            vcol = 'avg_travel_time'
            ploy_func = self._traj_ploy_features

        self.meta_cols = cols
        self.vcol = vcol
        self.ploy_func = ploy_func
