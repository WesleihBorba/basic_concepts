# Goal:

# Fazer um dataset com 3 grupos (pensar que grupos usar - sobre o que vai ser)

# Descobrir como encontrar o K corretamente e colocar em um DEF e mostrar gráficamente com dados reais para ver como fica a mudança em forma de circulo, como feito aqui: K-Nearest Neighbors Classifier

# PRECISA NORMALIZAR AS INFORMAÇÕES

classifier = KNeighborsClassifier(n_neighbors=3)

# An important thing about this algorithm is that we have to standardize each feature to have a mean zero and a variance one
# Olhar as assumptions tipo a anterior

