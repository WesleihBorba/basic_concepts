# Goal: Page recommendations based on conversion using A/B testing (Two-Tailed test)
import pandas as pd
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt
from scipy.stats import binom
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
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

# Hypothesis
null_hypothesis = 'Conversion A = Conversion B. Decision: Keep the current version (A)'
alternative_hypothesis = 'Conversion A <> Conversion B. Decision: Change to new version (B)'

# Effect size (Cohen's d): 0.2 (small), 0.5 (medium), 0.8 (large)
effect_size = 0.05  # The magnitude of the difference you want to detect.
alpha = 0.05  # Type I error probability
power = 0.80  # 80% chance of detecting the effect.


class ABTest:
    def __init__(self):
        self.data = pd.read_csv('files\\ab_data.csv')  # The data has already been randomized.
        self.group_control, self.group_treatment = (self.data[self.data['group'] == 'control'],
                                                    self.data[self.data['group'] == 'treatment'])

    def analyze_sample_size(self):
        logger.info('Sample size necessary to run A/B test')
        power_analysis = smp.TTestIndPower()

        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0,  # Ratio between size group 1/group 2
            alternative='two-sided'
        )

        logger.info(f"Required sample size per group: {round(sample_size)}")
        logger.info(f"Size of control group {len(self.group_control)}")
        logger.info(f"Size of treatment group {len(self.group_treatment)}")

    def exploratory_analysis(self):
        logger.info('Size Group X Converted Group')
        n_A = len(self.group_control)
        n_B = len(self.group_treatment)
        conv_A = self.group_control['converted'].sum()
        conv_B = self.group_treatment['converted'].sum()

        p_A = conv_A / n_A
        p_B = conv_B / n_B

        logger.info(f"Tax A: {p_A:.4f}, Tax B: {p_B:.4f}")

        logger.info('Null data?')
        null_group_control = self.group_control.isnull().sum().sum()
        null_group_treatment = self.group_treatment.isnull().sum().sum()
        if null_group_control or null_group_treatment > 0:
            logger.warning(f"Null data in any group: {null_group_control} or {null_group_treatment}")
        else:
            logger.debug('Not Null')

        logger.info('Duplicated data?')
        duplicated_group_control = self.group_control.duplicated().sum()
        duplicated_group_treatment = self.group_treatment.duplicated().sum()
        if duplicated_group_control or duplicated_group_treatment > 0:
            logger.warning(f"Duplicated data in any group: {duplicated_group_control} or {duplicated_group_treatment}")
        else:
            logger.debug('Not Duplicated')

        logger.info('Binomial Distribution - (focused on the media to be visible)')
        start = int(conv_A - 50)
        end = int(conv_A + 50)
        x = np.arange(start, end)
        pmf_A = binom.pmf(x, n_A, p_A)
        pmf_B = binom.pmf(x, n_B, p_B)

        plt.plot(x, pmf_A, label='Group A (Control)')
        plt.plot(x, pmf_B, label='Group B (Treatment)')
        plt.fill_between(x, pmf_A, alpha=0.2)
        plt.fill_between(x, pmf_B, alpha=0.2)
        plt.legend()
        plt.show()

    def run_hypothesis_test(self):
        logger.info('Running Hypothesis')
        counts = [self.group_treatment['converted'].sum(), self.group_control['converted'].sum()]
        nobs = [len(self.group_treatment), len(self.group_control)]

        stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')

        logger.info(f"Z-statistic: {stat:.4f}")
        logger.info(f"P-value: {p_value:.4f}")

        if p_value <= alpha:
            if stat > 0:  # B is better than A?
                logger.info(f"SUCCESS: B is significantly better than A. {alternative_hypothesis}")
            else:
                logger.info(f"DEFEAT: B is significantly worse than A. {null_hypothesis}")
        else:
            logger.info(f"INCONCLUSIVE: There is no evidence of a difference. {null_hypothesis}")


class_marketing = ABTest()
class_marketing.analyze_sample_size()
class_marketing.exploratory_analysis()
class_marketing.run_hypothesis_test()