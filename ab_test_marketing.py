# Goal: Page recommendations based on conversion using A/B testing (Two-Tailed test)
import pandas as pd
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt
from scipy.stats import binom
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
null_hypothesis = 'Conversion A = Conversion B'
alternative_hypothesis = 'Conversion A <> Conversion B'

# Effect size (Cohen's d): 0.2 (small), 0.5 (medium), 0.8 (large)
effect_size = 0.05  # The magnitude of the difference you want to detect.
alpha = 0.05  # Type I error probability
power = 0.80  # 80% chance of detecting the effect.


class ABTest:
    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\ab_data.csv')  # The data has already been randomized.
        self.group_control, self.group_treatment = (self.data[self.data['group'] == ['control']],
                                                    self.data[self.data['group'] == ['treatment']])

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
        logger.info('Null data?')
        logger.debug(self.group_control.isna())
        logger.debug(self.group_treatment.isna())

        logger.debug(self.group_control.isnull().sum())
        logger.debug(self.group_treatment.isnull().sum())

        logger.info('Binomial Distribution')
        pmf_A = binom.pmf(conversoes, n_A, p_A)
        pmf_B = binom.pmf(conversoes, n_B, p_B)

        # Plotar
        plt.figure(figsize=(10, 6))
        plt.bar(conversoes, pmf_A, label='Grupo A (Controle)', alpha=0.5)
        plt.bar(conversoes, pmf_B, label='Grupo B (Tratamento)', alpha=0.5)
        plt.xlabel('Número de Conversões')
        plt.ylabel('Probabilidade')
        plt.title('Distribuição Binomial: Grupo A vs Grupo B')
        plt.legend()
        plt.show()

        # Minimum conversion per group


#3. Análise Exploratória: Verifique se há usuários duplicados ou desequilíbrio entre os grupos.

#4. Aplicação do Teste: Utilize o Z-test para proporções (já que conversão é binária).

#5. Cálculo do Intervalo de Confiança: Essencial para mostrar a margem de erro da melhoria.

#6. Conclusão de Negócio: "Recomendamos implementar a nova página pois ela gera um aumento real de X% na conversão".

class_marketing = ABTest()
class_marketing.analyze_sample_size()
class_marketing.exploratory_analysis()