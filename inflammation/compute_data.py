"""Module containing mechanism for calculating standard deviation between datasets.
"""

import os
import glob
import numpy as np
from inflammation import models, views

class CSVDataSource:
    """
    Loads all the inflammation CSV files within a specified directory.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load_inflammation_data(self):
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation CSV files found in path {self.dir_path}")
        data = map(models.load_csv, data_file_paths)
        return list(data)

def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets."""
    data = data_source.load_inflammation_data()
    means_by_day = [models.daily_mean(dataset) for dataset in data]
    means_by_day_matrix = np.stack(means_by_day)
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process inflammation data.")
    parser.add_argument(
        'infiles', nargs='+', help="List of inflammation CSV files."
    )
    args = parser.parse_args()

    # Derive the directory from the first file in infiles
    dir_path = os.path.dirname(args.infiles[0])
    data_source = CSVDataSource(dir_path)
    analyse_data(data_source)
