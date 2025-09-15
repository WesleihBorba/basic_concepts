# Goal: Read any files with regular expression
import logging
import re


class ReadingFiles:

    def __init__(self):
        self.local_file = ('C:\\Users\\Weslei\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                           'Fases da vida\\Fase I\\Repository Projects\\files')

    def read_data(self):
        pass

    def regex_type(self):
        texto = "Olá, mundo! 12345 ABC def."

        # Encontrar todos os dígitos
        digitos = re.findall(r'\d', texto)
        print("Dígitos:", digitos)

        # Encontrar todos os caracteres que não são dígitos
        nao_digitos = re.findall(r'\D', texto)
        print("Não dígitos:", nao_digitos)

        # Encontrar todos os caracteres de palavra
        palavras = re.findall(r'\w', texto)
        print("Palavras:", palavras)

        # Encontrar todos os caracteres que não são de palavra
        nao_palavras = re.findall(r'\W', texto)
        print("Não palavras:", nao_palavras)

        # Encontrar todos os espaços em branco
        espacos = re.findall(r'\s', texto)
        print("Espaços em branco:", espacos)

        # Encontrar todos os caracteres que não são espaços em branco
        nao_espacos = re.findall(r'\S', texto)
        print("Não espaços em branco:", nao_espacos)


# PAREI AQUI: Grouping