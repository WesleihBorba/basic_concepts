# Goal: Page recommendations based on conversion using A/B testing
import pandas as pd


class ABTest:
    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\ab_data.csv')

        print(self.data)


class_marketing = ABTest()
