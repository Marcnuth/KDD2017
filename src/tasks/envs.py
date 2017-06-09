import luigi
import logging
from datetime import datetime
from pathlib import Path
import settings

logger = logging.getLogger('luigi-interface')
logger.setLevel('INFO')


class EnvSetUp(luigi.Task):

    unique_dirname = luigi.Parameter()
    #the timestamp must place here to keep unchanged in whole running time
    timestamp = datetime.today().date().isoformat() + '_'

    def output(self):
        dirname = self.timestamp + self.unique_dirname
        target = settings.results_dir / dirname
        return luigi.LocalTarget(target.absolute().as_posix())

    def run(self):
        target_dir = Path(self.output().path)
        if target_dir.exists():
            logger.info('Here')
            target_dir.rename(target_dir.absolute().as_posix() + '_bak')

        target_dir.mkdir()
