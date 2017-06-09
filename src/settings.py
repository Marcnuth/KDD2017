import os
from pathlib import Path as path

SETTING_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

project_dir = path(os.path.dirname(SETTING_BASE_DIR))

results_dir = project_dir / 'results'


class Data(object):

    submit_trajectories = path(SETTING_BASE_DIR + '/../data/dataSets_p1/submission_sample_travelTime.csv')
    submit_volume = path(SETTING_BASE_DIR + '/../data/dataSets_p1/submission_sample_volume.csv')

    class Test(object):
        trajectories = path(SETTING_BASE_DIR + '/../data/dataSets_p1/testing_phase1/trajectories_table5_test1.csv')
        volume = path(SETTING_BASE_DIR + '/../data/dataSets_p1/testing_phase1/volume_table6_test1.csv')
        weather = path(SETTING_BASE_DIR + '/../data/dataSets_p1/testing_phase1/weather_table7_test1.csv')

    class Train(object):
        links = path(SETTING_BASE_DIR + '/../data/dataSets_p1/training/links_table3.csv')
        routes = path(SETTING_BASE_DIR + '/../data/dataSets_p1/training/routes_table4.csv')
        trajectories = path(SETTING_BASE_DIR + '/../data/dataSets_p1/training/trajectories_table5_training.csv')
        volume = path(SETTING_BASE_DIR + '/../data/dataSets_p1/training/volume_table6_training.csv')
        weather = path(SETTING_BASE_DIR + '/../data/dataSets_p1/training/weather_table7_training.csv')


resources = {
    'train_trajectories': Data.Train.trajectories,
    'train_volume': Data.Train.volume,
    'valids_trajectories': Data.Test.trajectories,
    'valids_volume': Data.Test.volume
}
