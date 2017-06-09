import luigi
import settings
import logging
from common import aggregator
import pandas as pd
from pathlib import Path
from .envs import EnvSetUp
from common import utils


logger = logging.getLogger('luigi-interface')
logger.setLevel('INFO')


class Aggregator(luigi.Task):
    uuid = luigi.Parameter()
    resource = luigi.Parameter()

    def requires(self):
        return EnvSetUp(self.uuid)

    def output(self):
        errmsg = 'Allow resources:{0}'.format(settings.resources.keys())
        assert self.resource in settings.resources.keys(), errmsg

        suffix = 'agg_{0}.csv'.format(self.resource.lower())
        output_file = Path(self.input().path) / suffix
        return luigi.LocalTarget(output_file.absolute().as_posix())

    def run(self):

        (to_agg, func) = self._source()
        tmp = func(to_agg.absolute().as_posix())

        data = utils.convert(pd.DataFrame(tmp[1:], columns=tuple(tmp[0])))
        data.to_csv(self.output().path, index=False)

    def _source(self):

        rfname = settings.resources[self.resource].name

        is_traj = 'trajectories' in rfname
        is_train = 'train' in rfname

        if is_train and is_traj:
            return (settings.Data.Train.trajectories, aggregator.avgTravelTime)
        elif not is_train and is_traj:
            return (settings.Data.Test.trajectories, aggregator.avgTravelTime)
        elif is_train and not is_traj:
            return (settings.Data.Train.volume, aggregator.avgVolume)
        else:
            return (settings.Data.Test.volume, aggregator.avgVolume)
