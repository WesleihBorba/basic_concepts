# Goal: We will use logging to monitor the application of our cluster
import logging
import sys

# Logger setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(message)s]')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

# Class cluster
class DiabeticCluster:
    pass

# https://medium.com/@dKimothi/finding-best-features-for-predicting-diabetes-8656cf0d1185

# TRAINING LOGS


# EVALUATION LOGS
# PREDICTIONS LOGS
# SYSTEM LOGS (CPU AND MEMORY USAGE)

print(logger.addHandler(stream_handler))

logger.setLevel(logging.DEBUG)

logger.info("Converting from {from_country} to USD: {converted_to_usd}".format(from_country='from_country',
                                                                     converted_to_usd='converted_to_usd'))
logger.debug("Current rates: {exchange_rates}".format(exchange_rates='exchange_rates'))
logger.error("The TO country supplied is not a valid country.")
logger.log(logging.CRITICAL, 'Test critical')
logger.log(logging.INFO, "Test")
logger.warning('Values')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("my_program.log")
logger.addHandler(file_handler)

sys.exit(0)