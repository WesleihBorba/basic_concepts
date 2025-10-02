# Goal:
import numpy as np
import matplotlib.pyplot as plt


# Let's generate 10,000 "experiments"
# N = 10 shots
# P = 0.30 (30% he'll get a free throw)

a = np.random.binomial(10, 0.30, size=10000)
plt.hist(a, range=(0, 10), bins=10)
plt.xlabel('Number of "Free Throws"')
plt.ylabel('Frequency')
plt.show()