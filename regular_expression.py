# Goal: Read any files with regular expression
import logging
import re


class ReadingFiles:

    def __init__(self):
        self.local_file = ('C:\\Users\\Weslei\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                           'Fases da vida\\Fase I\\Repository Projects\\files')

    def read_data(self):
        pass

    def regex_shorthland(self):
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

    def regex_grouping(self):
        # String de exemplo
        texto = "A conferência será realizada no dia 25/06/2024 em São Paulo."

        # Padrão de expressão regular com agrupamento para dia, mês e ano
        padrao = r"(\d{2})/(\d{2})/(\d{4})"

        # Procurando o padrão na string
        resultado = re.search(padrao, texto)

        if resultado:
            # Usando grupos para capturar dia, mês e ano
            dia = resultado.group(1)
            mes = resultado.group(2)
            ano = resultado.group(3)

            print(f"Dia: {dia}")
            print(f"Mês: {mes}")
            print(f"Ano: {ano}")
        else:
            print("Data não encontrada na string.")


    def regex_quantifiers(self):
        # Texto de exemplo
        texto = "O preço é $100, $20, e $5."

        # Padrão para encontrar preços (um cifrão seguido de um ou mais dígitos)
        padrao_mais = r'\$\d+'

        # Padrão para encontrar números com pelo menos 2 dígitos
        padrao_min_dois = r'\b\d{2,}\b'

        # Padrão para encontrar números com exatamente 3 dígitos
        padrao_tres = r'\b\d{3}\b'

        # Encontrar todos os padrões no texto
        precos = re.findall(padrao_mais, texto)
        numeros_min_dois = re.findall(padrao_min_dois, texto)
        numeros_tres = re.findall(padrao_tres, texto)

        print("Preços encontrados:", precos)
        print("Números com pelo menos 2 dígitos:", numeros_min_dois)
        print("Números com exatamente 3 dígitos:", numeros_tres)


    def regex_anchors(self):
        # Exemplo de texto
        textos = [
            "Olá, Mundo!",
            "Olá a todos!",
            "Mundo, Olá!",
            "Boa tarde, Mundo!",
            "Olá, Mundo"
        ]

        # Expressão regular com âncoras
        pattern = r"^Olá.*Mundo$"

        # Função para verificar os textos
        def verificar_textos(textos):
            for texto in textos:
                if re.match(pattern, texto):
                    print(f"'{texto}' combina com a expressão '{pattern}'")
                else:
                    print(f"'{texto}' NÃO combina com a expressão '{pattern}'")

        # Chamar a função para verificar os textos
        verificar_textos(textos)



# PAREI AQUI: Grouping