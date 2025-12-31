# Goal: Create a Decision Trees, predict loan credit and plot our tree
import pandas as pd
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


class DecisionTreesClassification:

    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                                'Fases da vida\\Fase I\\Repository Projects\\files\\loan.csv') # Deixar apenas o arquivo

    def transforming_string_data(self):
        pass

classification = DecisionTreesClassification()


exit()

dt = DecisionTreeClassifier(max_depth=2, ccp_alpha=0.01,criterion='gini')


classifier.fit(training_data, training_labels)
predictions = classifier.predict(test_data)
print(classifier.score(test_data, test_labels))



# Olhar as assumptions of trees

# Criar o plot para a trees


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    for subset in split_labels:
      info_gain -= gini(subset) * len(subset) / len(starting_labels)
    return info_gain



# Usar debug no information gain ou Gini impurity
# Entender como é feito a separação de cada item nos nó. Por exemplo 2 na esquerda, 3 na direita
# Entender como interpretar decision tree
# Quais modelos de evaluation usar para o tipo de dados que eu usar
# Entender o que é o Depth max
# Entender e usar o pruning
# Como interpretar o resultado da Decision Trees, colocar isso no arquivo também


## https://www.kaggle.com/datasets/sujithmandala/simple-loan-classification-dataset/data
## https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling