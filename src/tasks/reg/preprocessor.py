import luigi
import pandas as pd
import numpy as mp
from sklearn import preprocessing
from tasks.reg.combiner import CombineFeatures
from pathlib import Path
from common import utils
from sklearn import decomposition
from sklearn import pipeline
from sklearn import feature_selection


class Preprocess(luigi.Task):

    uuid = luigi.Parameter()
    task = luigi.Parameter()
    pipe = luigi.Parameter(default='poly,std,dec')

    def requires(self):
        return CombineFeatures(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        fname = 'preprocess_' + Path(self.input().path).name
        outfile = self.dir / fname
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def _process(self, data):
        #features = preprocessing.PolynomialFeatures().fit_transform(features)
        features = preprocessing.RobustScaler().fit_transform(data)
        #features = decomposition.TruncatedSVD().fit_transform(features)
        
        #cols = list(['f_' + i for i in range(features.shape[1])])
        return pd.DataFrame(features, columns=data.columns)

    def run(self):
        self._source()

        df = utils.load_data(self.input().path)

        final = pd.DataFrame([])
        for g in df.groupby(self.key_cols):

            (keys, data) = g

            meta_cols = data[self.meta_cols]
            processed_df = self._process(data.drop(self.meta_cols, axis=1))

            # reset the index, otherwise the concat will not work
            meta_cols.reset_index(drop=True, inplace=True)
            processed_df.reset_index(drop=True, inplace=True)
            g_final = pd.concat([meta_cols, processed_df], axis=1)

            final = pd.concat([final, g_final])

        #features = feature_selection.VarianceThreshold().fit_transform(features)

        #print(ployed_df.head())
        #print(meta_df.head())
        #print(final.head())

        final.to_csv(self.output().path, index=False)

    def _source(self):
        time_cols = ['time_window_start', 'time_window_end']
        if self.task == 'volume':
            keycols = ['tollgate_id', 'direction']
            vcol = 'volume'
        else:
            keycols = ['intersection_id', 'tollgate_id']
            vcol = 'avg_travel_time'

        self.key_cols = keycols
        self.meta_cols = [*time_cols, *keycols, vcol]
        self.vcol = vcol
