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
from tasks.tsa.model import TSARIMA
from tasks.reg.skmodels import SkAlgorithm, Voter


logger = logging.getLogger('luigi-interface')
logger.setLevel('INFO')


class Summary(luigi.Task):
    uuid = luigi.Parameter()
    task = luigi.Parameter()
    algorithm = luigi.Parameter()

    def requires(self):
        reqs = []

        if self.algorithm.lower() == 'arima':
            reqs.append(TSARIMA(self.uuid, self.task))
        elif self.algorithm.lower().startswith('sk-'):
            reqs.append(SkAlgorithm(self.uuid, self.task, self.algorithm[3:]))
        elif self.algorithm.lower() == 'vote':
            reqs.append(Voter(self.uuid, self.task))
        else:
            raise Exception('Algorithm options: ARIMA, SK-*')

        if self.task == 'trajectories':
            reqs.append(Aggregator(self.uuid, 'valids_trajectories'))
        else:
            reqs.append(Aggregator(self.uuid, 'valids_volume'))

        return reqs

    def output(self):
        self.dir = Path(self.input()[1].path).absolute().parent
        outfile = self.dir / 'summary_{0}_{1}.org'.format(self.task, self.algorithm)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):

        self.valids = utils.load_data(self.input()[0][0].path)
        self.tests = utils.load_data(self.input()[0][1].path)
        self.valids_real = utils.load_data(self.input()[1].path)

        utils.valid_submssion(self.task, self.tests)

        (df, mape) = self._calculate_mape()
        fig = self._output_plots(df)

        with open(self.output().path, 'w') as f:
            f.write('MAPE for {0} = {1}\n[[file:{2}][Lines compare]]\n'.format(
                self.task, mape, fig
            ))

        logger.info('===== MAPE = {0} for {1} ======'.format(mape, self.task))

    def _calculate_mape(self):

        # read predicts & real
        forecast = self.valids
        forecast['time_window_start'] = forecast.time_window.map(
            lambda x: x.split(',')[0][1:]
        )
        forecast['time_window_start'] = pd.to_datetime(forecast.time_window_start)

        real = self.valids_real

        if self.task == 'trajectories':
            onkeys = ['intersection_id', 'tollgate_id', 'time_window_start']
        else:
            onkeys = ['tollgate_id', 'direction', 'time_window_start']

        final = pd.merge(forecast, real, how='inner', on=onkeys)
        final.sort(['time_window_start'], inplace=True)

        mape = []

        if self.task == 'trajectories':
            vcol = 'avg_travel_time'
            groupbys = [['intersection_id'], ['tollgate_id']]
        else:
            vcol = 'volume'
            groupbys = [['tollgate_id'], ['direction']]

        for intersections in final.groupby(groupbys[0]):
            for route in intersections[1].groupby(groupbys[1]):
                pred, real = route[1][vcol + '_x'], route[1][vcol + '_y']
                ratio = abs(real - pred) / real
                mape.append(ratio.mean())

        return (final, np.average(mape))

    def _output_plots(self, final):

        if self.task == 'trajectories':
            vcol = 'avg_travel_time'
        else:
            vcol = 'volume'

        plt.figure()
        sns.tsplot(final[vcol + '_x'], color='green',
                   condition='prediction', legend=True, linewidth=2)
        sns.tsplot(final[vcol + '_y'], color='grey',
                   condition='actual', legend=True, linewidth=1)
        figfile = self.dir / 'summary_valids_lines_{0}_{1}.jpg'.format(
            self.task, self.algorithm)
        plt.savefig(figfile.absolute().as_posix())

        return figfile.absolute().as_posix()

