import pandas as pd
import numpy as np
from common import utils
import luigi
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import preprocessing, decomposition
from tasks.reg.polys import PolyFeatures


class GBDTFeatures(luigi.Task):
    '''
    More information about Decision Tree, please refer to:
    http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    '''

    uuid = luigi.Parameter()
    task = luigi.Parameter()
    n_tree = luigi.IntParameter(default=10)

    def requires(self):
        return PolyFeatures(self.uuid, self.task)

    def output(self):
        self.dir = Path(self.input().path).absolute().parent
        outfile = self.dir / 'feature_gbdt_{0}_{1}.csv'.format(
            self.task, self.n_tree)
        return luigi.LocalTarget(outfile.absolute().as_posix())

    def run(self):
        self._source()

        pool = utils.load_data(self.input().path)

        features = self._gen_gbdt_features(pool)
        features.to_csv(self.output().path, index=False)

    def _gen_gbdt_features(self, pool):
        final = []
        col_gbdts = ['gbdt_' + str(i) for i in range(self.n_tree)]
        for g in pool.groupby(self.kcols):
            (keys, df) = g

            train = df[df.time_window_start < pd.datetime(2016, 10, 18)]
            train_x = train.drop([*self.timecols, *self.kcols, self.vcol], axis=1)
            train_y = train[self.vcol]

            regor = GradientBoostingRegressor(
                loss='huber',
                n_estimators=self.n_tree, max_leaf_nodes=10)
            model = regor.fit(train_x, train_y)

            gbdt_features = []
            df_x = df.drop([*self.timecols, *self.kcols, self.vcol], axis=1)
            assert len(model.estimators_) == self.n_tree, 'n_tree is not match!'
            for tree in model.estimators_:
                gbdt_features.append(tree[0].apply(df_x))

            gbdt_features = np.array(gbdt_features).T
            df_gbdts = pd.DataFrame(gbdt_features, columns=col_gbdts)

            # ignore_index will drop column names, so we do not use it here
            df.reset_index(drop=True, inplace=True)
            df_gbdts.reset_index(drop=True, inplace=True)
            final.append(pd.concat([df, df_gbdts], axis=1))

        final = pd.concat(final)

        # one-hot code
        origs = final[list(set(final.columns) - set(col_gbdts))]
        gbdts = final[col_gbdts]

        for gbdtcol in col_gbdts:
            gbdts[gbdtcol] = gbdts[gbdtcol].astype('category')

        gbdts_dummies = pd.get_dummies(gbdts).reset_index(drop=True)

        n_feature = int(origs.shape[1] / 4)

        # remove duplicated gdbt features
        gbdts = gbdts_dummies.T.drop_duplicates().T
        #gbdts = preprocessing.RobustScaler().fit_transform(gbdts)
        #gbdts = decomposition.TruncatedSVD(
        #    n_components=int(origs.shape[1] / 4), n_iter=20
        gbdts = decomposition.KernelPCA(
            n_components=n_feature
        ).fit_transform(gbdts)
        gbdts = pd.DataFrame(
            gbdts, columns=['gbdt_{0}'.format(i) for i in range(n_feature)])

        print('GDBT Feature Shape:', gbdts.shape, origs.shape)

        origs.reset_index(drop=True, inplace=True)
        gbdts.reset_index(drop=True, inplace=True)

        return pd.concat([origs, gbdts], axis=1)

    def _source(self):
        if self.task == 'volume':
            kcols = ['tollgate_id', 'direction']
            vcol = 'volume'

        else:
            kcols = ['intersection_id', 'tollgate_id']
            vcol = 'avg_travel_time'

        self.timecols = ['time_window_start', 'time_window_end']
        self.kcols = kcols
        self.vcol = vcol
