import luigi
import settings
import pandas as pd
from pathlib import Path
from .envs import EnvSetUp
from common import utils
from datetime import timedelta, datetime
import math
import numpy as np
from functools import reduce


class ProcessLinks(luigi.Task):
    uuid = luigi.Parameter()

    def requires(self):
        return EnvSetUp(self.uuid)

    def output(self):
        suffix = 'processed_links.csv'
        output_file = Path(self.input().path) / suffix
        return luigi.LocalTarget(output_file.absolute().as_posix())

    def run(self):
        links = utils.load_data(settings.Data.Train.links)
        links['capacity'] = links.length * links.width
        links['tan'] = links.length / links.width

        links['intop_cnt'] = links.in_top.map(lambda x: len(str(x).split(',')))
        links['outtop_cnt'] = links.out_top.map(lambda x: len(str(x).split(',')))
        links['io_link_ratio'] = links.intop_cnt / links.outtop_cnt

        links['in_cap_ratio'], links['out_cap_ratio'] = np.nan, np.nan
        links['in_lane_ratio'], links['out_lane_ratio'] = np.nan, np.nan
        links['io_cap_ratio'], links['io_lane_ratio'] = np.nan, np.nan
        for row in links.iterrows():
            (index, data) = row

            intop = [] if pd.isnull(data.in_top) else data.in_top.split(',')
            outop = [] if pd.isnull(data.out_top) else data.out_top.split(',')

            in_caps, in_lanes = 0, 0
            for item in intop:
                in_caps += links[links.link_id == int(item)].capacity.iloc[0]
                in_lanes += links[links.link_id == int(item)].lanes.iloc[0]

            out_caps, out_lanes = 0, 0
            for item in outop:
                out_caps += links[links.link_id == int(item)].capacity.iloc[0]
                out_lanes += links[links.link_id == int(item)].lanes.iloc[0]

            # fix zero
            in_caps = in_caps or 1
            out_caps = out_caps or 1
            in_lanes = in_lanes or 1
            out_lanes = out_lanes or 1

            links.set_value(index, 'in_cap_ratio', float(in_caps / data.capacity))
            links.set_value(index, 'out_cap_ratio', float(data.capacity / out_caps))
            links.set_value(index, 'io_cap_ratio', float(in_caps / out_caps))

            links.set_value(index, 'in_lane_ratio', float(in_lanes / data.lanes))
            links.set_value(index, 'out_lane_ratio', float(data.lanes / out_lanes))
            links.set_value(index, 'io_lane_ratio', float(in_lanes / out_lanes))

        links.drop(['in_top', 'out_top', 'lane_width'], axis=1, inplace=True)
        links.to_csv(self.output().path, index=False)


class FlatData(luigi.Task):

    uuid = luigi.Parameter()
    resource = luigi.Parameter()

    def requires(self):
        return EnvSetUp(self.uuid)

    def output(self):
        errmsg = 'Allow resources:{0}'.format(settings.resources.keys())
        assert self.resource in settings.resources.keys(), errmsg

        suffix = 'flatten_{0}.csv'.format(self.resource.lower())
        output_file = Path(self.input().path) / suffix
        return luigi.LocalTarget(output_file.absolute().as_posix())

    def run(self):
        self._source()

        final = self.flat_func()
        final.to_csv(self.output().path, index=False)

    def _source(self):
        self.resource_file = settings.resources[self.resource].absolute().as_posix()
        if 'trajectories' in self.resource:
            self.flat_func = self._flat_trajectories
        else:
            raise Exception('Not finished')

    def _flat_trajectories(self):
        df = utils.load_data(self.resource_file)
        df.travel_time = pd.to_numeric(df.travel_time)
        df.vehicle_id = pd.to_numeric(df.vehicle_id)
        df.starting_time = pd.to_datetime(df.starting_time)

        # flat the source data
        flatted = []
        for row in df.iterrows():
            (index, data) = row
            total_cost = reduce(
                lambda x, y: x + y,
                map(lambda x: float(x.split('#')[-1]), data.travel_seq.split(';')))

            time_window_start = datetime(
                *data.starting_time.timetuple()[:4],
                math.floor(data.starting_time.minute / 20) * 20
            )
            time_window_end = time_window_start + timedelta(minutes=20)
            flatted.append([
                data.intersection_id, data.tollgate_id,
                time_window_start, time_window_end,
                data.travel_time, data.vehicle_id,
                total_cost, data.travel_time - total_cost
            ])

        flatted = pd.DataFrame(flatted, columns=(
            'intersection_id', 'tollgate_id', 'time_window_start', 'time_window_end',
            'travel_time', 'vehicle_id', 'cost', 'extracost'))

        return flatted


class MergeData(luigi.Task):

    uuid = luigi.Parameter()
    resource = luigi.Parameter()

    def requires(self):
        return [FlatData(self.uuid, self.resource), ProcessLinks(self.uuid)]

    def output(self):
        errmsg = 'Allow resources:{0}'.format(settings.resources.keys())
        assert self.resource in settings.resources.keys(), errmsg

        suffix = 'merge_{0}.csv'.format(self.resource.lower())
        output_file = Path(self.input()[0].path).absolute().parent / suffix
        return luigi.LocalTarget(output_file.absolute().as_posix())

    def run(self):
        self._source()

        final = self.flat_func()
        final.to_csv(self.output().path, index=False)

    def _source(self):
        self.resource_file = settings.resources[self.resource].absolute().as_posix()
        if 'trajectories' in self.resource:
            self.flat_func = self.travel_time_features
            self.trajectories = utils.load_data(self.input()[0].path)
        else:
            raise Exception('Not finished')

        self.links = utils.load_data(self.input()[1].path)

    def travel_time_features(self):

        #For travel time prediction, we predict for every single route
        #so the link data is not useful here, coz they keeps the same for one route
        # here we will only aggregate some info from the trajectories self.
        agged = []
        for g in self.trajectories.groupby(['intersection_id',
                                            'tollgate_id',
                                            'time_window_start']):

            ((iid, tid, twstart), data) = g

            vehicles = len(set(data.vehicle_id))
            twend = twstart + timedelta(minutes=20)

            agged.append([
                iid, tid, twstart, twend, data.travel_time.mean(),
                0 if pd.isnull(data.travel_time.std()) else data.travel_time.std(),
                vehicles, data.extracost.sum(), data.extracost.median()
            ])

        final = pd.DataFrame(agged, columns=[
            'intersection_id', 'tollgate_id', 'time_window_start',
            'time_window_end', 'avg_travel_time', 'travel_time',
            'vehicles', 'extracost1', 'extracost2'
        ])
        return final
