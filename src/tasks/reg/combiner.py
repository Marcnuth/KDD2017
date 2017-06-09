import luigi
import pandas as pd
from tasks.reg.features import ARIMAFeatures
from pathlib import Path
from common import utils
from tasks.reg.gbdt import GBDTFeatures


class CombineFeatures(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()

    def requires(self):
        return [GBDTFeatures(self.uuid, self.task),
                ARIMAFeatures(self.uuid, self.task)]

    def output(self):
        self.dir = Path(self.input()[0].path).absolute().parent
        outfile = self.dir / 'combined_{0}.csv'.format(self.task)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        df0 = utils.load_data(self.input()[0].path)

        df1 = utils.load_data(self.input()[1].path)

        final = pd.merge(df0, df1, how='left', on=self.oncols)

        # excludes those data in col:tsa from ARIMA which is null
        final = final[-final.tsv.isnull()]
        final.to_csv(self.output().path, index=False)

    def _source(self):
        if self.task == 'volume':
            oncols = ['tollgate_id', 'direction', 'time_window_start']
        else:
            oncols = ['intersection_id', 'tollgate_id', 'time_window_start']

        self.oncols = oncols

