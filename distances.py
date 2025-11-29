# Goal:
from scipy.spatial import distance



# Representing Points: types of distance

## Euclidean Distance -- No cálculo de vizinhos (KNN) -- Na predição
print(distance.euclidean([1, 2], [4, 0]))

## Manhattan Distance -- Lasso Regression -- Na função de custo quando usa penalização L1
print(distance.cityblock([1, 2], [4, 0]))

## Hamming Distance -- K-Modes (clustering para dados categóricos)
print(distance.hamming([5, 4, 9], [1, 7, 9]))

## Temtar usar para um exemplo assim: Distance functions are often used as error or cost functions to be minimized in an optimization problem.

