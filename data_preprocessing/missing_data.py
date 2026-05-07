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
        logger.info('Reading a missing file data')
        self.data_missing = pd.read_csv('files\\missing_file.csv', sep=';')

    def locf_missing(self):
        logger.info('Add last observation carried forward')

        logger.info(f'Before to change our data: {self.data_missing['value_stock']}')
        self.data_missing['value_stock'] = self.data_missing['value_stock'].ffill(axis=0)
        logger.info(f'Data after change: {self.data_missing}')

    def nocb_missing(self):
        logger.info('Add next observation carried backward')

        logger.info(f'Before to change our data: {self.data_missing['maximum_stock']}')
        self.data_missing['maximum_stock'] = self.data_missing['maximum_stock'].bfill(axis=0)
        logger.info(f'Data after change: {self.data_missing}')

    def other_methods(self):
        logger.info('Filling with the first value of our dataframe')
        logger.info(f'Before to change our data: {self.data_missing['volume']}')
        first_value = self.data_missing['volume'][0]
        self.data_missing['volume'] = self.data_missing['volume'].fillna(value=first_value)
        logger.info(f'Data after change: {self.data_missing}')

        logger.info('Filling with the worst value, minimal ou maximum, depends of our data')
        worst = self.data_missing['minimum_stock'].min()
        logger.info(f'Before to change our data: {self.data_missing['minimum_stock']}')
        self.data_missing['minimum_stock'] = self.data_missing['minimum_stock'].fillna(value=worst)
        logger.info(f'Data after change: {self.data_missing}')

    def drop_missing_data(self):
        logger.info('Dropping missing data, the last alternative')

        logger.info(f'Before to change our data: {self.data_missing}')
        self.data_missing.dropna(inplace=True)
        logger.info(f'Data after change: {self.data_missing}')


new_class = MissingData()
new_class.locf_missing()
new_class.nocb_missing()
new_class.other_methods()
new_class.drop_missing_data()
