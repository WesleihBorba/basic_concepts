# Learn how to use Dunder Methods
import numpy as np


class AnalyzeArrays:

    def __init__(self, data):  # Initialize the values
        self.array_origin = np.asarray(data)
        self.transformed_array = []

    def __add__(self, other):  # Increase the values in a new array
        if isinstance(other, AnalyzeArrays):
            if self.array_origin.shape != other.array_origin.shape:
                raise ValueError("Arrays with different shapes cannot be added.")
            return AnalyzeArrays(self.array_origin + other.array_origin)
        else:
            return AnalyzeArrays(self.array_origin + other)

    def __len__(self):  # Size of each data
        return len(self.array_origin)

    def __eq__(self, other):  # Know if the values are equals
        if not isinstance(other, AnalyzeArrays):
            return NotImplemented
        return np.array_equal(self.array_origin, other.array_origin)

    def __repr__(self) -> str:  # Show the values
        return f"{self.__class__.__name__}({self.array_origin!r})"


array_1 = np.array([2, 5, 6])
array_2 = np.array([8, 3, 1])

analyze_1 = AnalyzeArrays(array_1)
analyze_2 = AnalyzeArrays(array_2)

sum_array = analyze_1 + analyze_2
print(repr(sum_array))

equals_values = analyze_1 == analyze_2
size_values = len(analyze_1) == len(analyze_2)
print(repr(equals_values))
print(repr(size_values))