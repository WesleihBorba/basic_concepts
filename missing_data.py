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
        pass

    def locf_missing(self):
        logger.info('Add last observation carried forward')
        self.data_missing['comfort'].ffill(axis=0, inplace=True)


    def drop(self):
        self.data_missing.dropna(inplace=True)

# PAREI AQUI: Missing at Random (MAR)