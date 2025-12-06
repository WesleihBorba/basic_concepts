# Goal: Understanding types of normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Normalization:

    def __init__(self):
        self.skew_data = pd.DataFrame({
            "exp_skew": np.random.exponential(scale=10, size=1000),
            "log_skew": np.random.lognormal(mean=2, sigma=0.7, size=1000),
            "gamma_skew": np.random.gamma(shape=2, scale=3, size=1000)
        })

    @staticmethod
    def plot_comparison(original_df, transformed_df, title):
        num_cols = len(original_df.columns)
        plt.figure(figsize=(15, 4 * num_cols))

        for i, col in enumerate(original_df.columns):
            plt.subplot(num_cols, 2, 2*i + 1)
            plt.hist(original_df[col], bins=40)
            plt.title(f"{col} – Before ({title})")

            plt.subplot(num_cols, 2, 2*i + 2)
            plt.hist(transformed_df[col], bins=40)
            plt.title(f"{col} – After ({title})")

        plt.tight_layout()
        plt.show()

    def min_max(self):
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(self.skew_data), columns=self.skew_data.columns)
        return data

    def z_score(self):
        zscore = StandardScaler()
        data = pd.DataFrame(zscore.fit_transform(self.skew_data), columns=self.skew_data.columns)
        return data

    def log_transformation(self):
        data = pd.DataFrame(np.log(self.skew_data))
        return data


norm_class = Normalization()

# Min-Max
data_minmax = norm_class.min_max()
norm_class.plot_comparison(norm_class.skew_data, data_minmax, title="Min-Max")

# Z-score
data_zscore = norm_class.z_score()
norm_class.plot_comparison(norm_class.skew_data, data_zscore, title="Z-score")

# Log Transformation
data_log = norm_class.log_transformation()
norm_class.plot_comparison(norm_class.skew_data, data_log, title="Log Transform")
