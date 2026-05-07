# Goal: Analyze whether the marketing campaign was a success.
import numpy as np
import pandas as pd
from dowhy import CausalModel
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

np.random.seed(42)
N = 5000

logger.info('Confounders variables, influence who receives the AD.')
age = np.random.normal(35, 10, N)
previous_expense = np.random.gamma(2, 50, N)

logger.info('Treatment: Our action - what we will measure, who receive AD')
prob_ad = 1 / (1 + np.exp(-(previous_expense - 100) / 20))
ad = np.random.binomial(1, prob_ad)  # Probability of receiving an ad (marketing) with selection bias

logger.info('Results of campaign, how much we gain with AD')
actual_sell = (50 + (20 * ad) + (0.4 * previous_expense) +
               np.random.normal(0, 5, N))  # "20" real effect - this values our model need to find

df = pd.DataFrame({
    'ad_marketing': ad.astype(bool),
    'actual_sell': actual_sell,
    'age': age,
    'previous_expense': previous_expense
})

logger.info('Define who is the causal of our sells')
model = CausalModel(
    data=df,
    treatment='ad_marketing',
    outcome='actual_sell',
    common_causes=['age', 'previous_expense']  # Confounders
)

logger.info('DoWhy finds the statistical formula to isolate the effect')
identified_estimated = model.identify_effect(proceed_when_unidentifiable=True)

logger.info('Use our previous model to estimate effect; Normally use linear regression or propensity score')
estimate = model.estimate_effect(
    identified_estimated,
    method_name="backdoor.linear_regression"
)

logger.debug(f"Estimate Effect (ATE - Average Treatment Effect): {estimate.value}")

logger.info('Stress test, add a random common cause, ATE can not change a lot - validation of model')
refutation = model.refute_estimate(
    identified_estimated,
    estimate,
    method_name="random_common_cause"
)

logger.debug(refutation)