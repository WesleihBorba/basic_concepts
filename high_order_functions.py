import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating a dataset
np.random.seed(42)
mu, sigma = 3, 1
normal_random = np.random.normal(mu, sigma, 150)

features = ['Omega', 'Sigma', 'Beta']
dataset_1, dataset_2, dataset_3 = [normal_random[i:i + 50] for i in range(0, len(normal_random), 50)]

dataset_1 = pd.DataFrame({'value': dataset_1, 'feature': features[0]})
dataset_2 = pd.DataFrame({'value': dataset_2, 'feature': features[1]})
dataset_3 = pd.DataFrame({'value': dataset_3, 'feature': features[2]})

full_data = pd.concat([dataset_1, dataset_2, dataset_3])

plt.hist(full_data['value'])
plt.show()

# using map function to normalize to 0 and 1
max_data = full_data['value'].max(axis=0)
min_data = full_data['value'].min(axis=0)
full_data['value'] = list(map(lambda data: ((data-min_data) / (max_data-min_data)) * (1-0) + 0,
                              full_data['value']))


plt.hist(full_data['value'])
plt.show()
