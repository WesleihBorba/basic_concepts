# Goal: Treat data with missing values
import pandas as pd
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class MissingData:

    def __init__(self):
        self.data_missing = pd.DataFrame()

    def read_file(self):
        logger.info('Reading a missing file data')

    def locf_missing(self):
        logger.info('Add last observation carried forward')
        self.data_missing['comfort'].ffill(axis=0, inplace=True)

    def nocb_missing(self):
        logger.info('Add next observation carried backward')
        self.data_missing['comfort'].bfill(axis=0, inplace=True)

    def other_methods(self):
        logger.info('Filling with the first value of our dataframe')
        first_value = self.data_missing['data'][0]
        self.data_missing['concentration'].fillna(value=first_value, inplace=True)

        logger.info('Filling with the worst value, minimal ou maximum, depends of our data')
        worst = self.data_missing['pain'].max()
        self.data_missing['pain'].fillna(value=worst, inplace=True)



    def drop(self):
        self.data_missing.dropna(inplace=True)

# PAREI AQUI: Missing at Random (MAR)