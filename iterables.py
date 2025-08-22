# Iterable in big data
import itertools
from sklearn.datasets import load_iris
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target


class ReadIterableBigData:  # Using yield to see data and getting one part to divide in train test
    def __init__(self, train_ratio=0.7):
        # Creating a big data
        self.x_data = np.tile(iris.data, (7000, 1))  # 150 * 7000 ≈ 1 million
        self.y_data = np.tile(iris.target, 7000)
        self.train_size = int(len(self.x_data) * train_ratio)

    def __iter__(self):
        # Divide in train and test
        for idx in range(len(self.x_data)):
            part_data = 'train' if idx < self.train_size else 'test'
            yield self.x_data[idx], self.y_data[idx], part_data


class TrainTestSplitData:  # Use Yield to read
    def __init__(self, source_iterable):
        self.source = source_iterable

    def __iter__(self):
        yield from self.source  # Use yield from to get the origin of data


data_reader = ReadIterableBigData()
split_data = TrainTestSplitData(data_reader)

train_count, test_count = 0, 0

for x, y, part in split_data:
    if part == 'train':
        train_count += 1
    else:
        test_count += 1

print(f"Train: {train_count}, Test: {test_count}")
