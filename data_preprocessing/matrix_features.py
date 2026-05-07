# Goal: Creating variables on a Matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, binom
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

n = 30000

logger.info('Normal distribution with an average around zero')
X1 = np.random.normal(0, 1, n)

logger.info('Frequency of events occurring within a fixed interval, using 3 as the frequency')
X2 = poisson.rvs(mu=3, size=n)

logger.info('Binomial distribution with 10 experiments independently')
X3 = binom.rvs(n=10, p=0.4, size=n)

logger.info('Uniform Distribution,generates continuous values [-1, 1] where all numbers within the interval have the '
            'same probability of occurring. ')
X4 = np.random.uniform(-1, 1, n)

logger.info('Category distribution, selects elements from a list with specific probabilities')
X5 = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])

logger.info('Creating matrix data')
X = np.column_stack([X1, X2, X3, X4, X5])

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

axs[0, 0].hist(X1, bins=30, color='skyblue', edgecolor='black')
axs[0, 0].set_title("X1: Normal")

unique2, counts2 = np.unique(X2, return_counts=True)
axs[0, 1].bar(unique2, counts2, color='salmon', edgecolor='black')
axs[0, 1].set_title("X2: Poisson")

unique3, counts3 = np.unique(X3, return_counts=True)
axs[0, 2].bar(unique3, counts3, color='lightgreen', edgecolor='black')
axs[0, 2].set_title("X3: Binomial")

axs[1, 0].hist(X4, bins=30, color='gold', edgecolor='black')
axs[1, 0].set_title("X4: Uniform")

unique5, counts5 = np.unique(X5, return_counts=True)
axs[1, 1].bar(unique5, counts5, color='orchid', edgecolor='black')
axs[1, 1].set_xticks([0, 1, 2])
axs[1, 1].set_title("X5: Choice")

fig.delaxes(axs[1, 2])
plt.show()