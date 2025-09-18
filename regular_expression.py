# Goal: Read files and extract information with regular expression
import logging
import sys
import re
import pandas as pd
import glob

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ReadingFiles:

    def __init__(self):
        self.local_file = ('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                           'Fases da vida\\Fase I\\Repository Projects\\files\\')

    def read_data(self, file, separation):
        logger.info('Reading a file')
        data = pd.read_csv(f'{self.local_file}{file}', sep=separation)
        return data

    def regex_shorthand(self):
        file_1 = self.read_data('file_1.csv', separation=';')

        logger.info('Find every number in CSV')
        numbers = file_1.dropna().map(lambda x: re.findall(r"\d+", x))
        numbers_list = [num for sublist in numbers.values.flatten() for num in sublist]
        logger.info(f"Numbers: {numbers_list}")

        logger.info('All text in that file')
        text = file_1.dropna().map(lambda x: re.findall(r"\D+", x))
        text_list = [num for sublist in text.values.flatten() for num in sublist]
        logger.info(f"Texts: {text_list}")

        logger.info('All characters but it is not a word')
        characters = file_1.dropna().map(lambda x: re.findall(r"\W+", x))
        characters_list = [num for sublist in characters.values.flatten() for num in sublist]
        logger.info(f"Not Words: {characters_list}")

    @staticmethod
    def regex_grouping():
        pattern_day = r"(\d{2})/(\d{2})/(\d{4})"
        pattern_place = r"in\s+([A-Za-z]+)"
        meeting_text = "The conference will be 25/06/2025 in London"

        logger.info('Find the day that we will occur our meeting, grouping per day, month and year')
        results = re.search(pattern_day, meeting_text)
        if results:
            day = results.group(1)
            month = results.group(2)
            year = results.group(3)

            logger.info(f"day: {day}")
            logger.info(f"Month: {month}")
            logger.info(f"Year: {year}")
        else:
            logger.info("There is no meeting")

        logger.info("Find a place will occur our meeting")
        place_result = re.search(pattern_place, meeting_text)
        if place_result:
            place = place_result.group(1)
            logger.info(f"Place: {place}")
        else:
            logger.info("There is no meeting")

    def regex_quantifiers(self):
        file_2 = self.read_data('file_2.csv', separation=';')

        logger.info('Find all values with 6 digits')
        stacked = file_2.drop(columns="account").stack()
        numbers_above_6_digits = stacked.map(
            lambda x: re.findall(r'\b\d{6}\b', str(int(float(x)))) if pd.notna(x) else []
        )

        # Getting index
        matches = numbers_above_6_digits[numbers_above_6_digits.map(len) > 0]

        for (line, columns), valor in matches.items():
            logger.info(f"Descriptions: {file_2.loc[line, 'account']}, {columns}, {valor}")

    def regex_read_all_data(self):
        logger.info('Reading all files')
        files = glob.glob(f'{self.local_file}*.csv')

        dataframes = []

        for f in files:
            try:
                df = pd.read_csv(f, sep=";")
                logger.info(f"Success using sep ; -> {f}")
            except Exception as e1:
                logger.info(f'Error {e1}')
                try:
                    df = pd.read_csv(f, sep=",")
                    logger.info(f"Success -> {f}")
                except Exception as e2:
                    logger.error(f"Error to read {f} with both separations: {e2}")
                    df = None

            if df is not None:
                dataframes.append(df)


ReadingFiles().regex_shorthand()
ReadingFiles().regex_grouping()
ReadingFiles().regex_quantifiers()
ReadingFiles().regex_read_all_data()
