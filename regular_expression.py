# Goal: Read files and extract information with regular expression
import logging
import sys
import re
import pandas as pd

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ReadingFiles:

    def __init__(self):
        self.local_file = ('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\'
                           'Fases da vida\\Fase I\\Repository Projects\\files\\')

    def read_data(self, file, separation):
        logger.info('Reading a file')
        data = pd.read_csv(f'{self.local_file}{file}', sep=separation)
        return data

    def regex_shorthand(self):
        file_1 = self.read_data('file_1.csv', separation=';')

        logger.info('Find every number in CSV')
        numbers = file_1.dropna().map(lambda x: re.findall(r"\d+", x))
        numbers_list = [num for sublist in numbers.values.flatten() for num in sublist]
        logger.info(f"Numbers: {numbers_list}")

        logger.info('All text in that file')
        text = file_1.dropna().map(lambda x: re.findall(r"\D+", x))
        text_list = [num for sublist in text.values.flatten() for num in sublist]
        logger.info(f"Texts: {text_list}")

        logger.info('All characters but it is not a word')
        characters = file_1.dropna().map(lambda x: re.findall(r"\W+", x))
        characters_list = [num for sublist in characters.values.flatten() for num in sublist]
        logger.info(f"Not Words: {characters_list}")

    @staticmethod
    def regex_grouping():
        pattern_day = r"(\d{2})/(\d{2})/(\d{4})"
        pattern_place = r"in\s+([A-Za-z]+)"
        meeting_text = "The conference will be 25/06/2025 in London"

        logger.info('Find the day that we will occur our meeting, grouping per day, month and year')
        results = re.search(pattern_day, meeting_text)
        if results:
            day = results.group(1)
            month = results.group(2)
            year = results.group(3)

            logger.info(f"day: {day}")
            logger.info(f"Month: {month}")
            logger.info(f"Year: {year}")
        else:
            logger.info("There is no meeting")

        logger.info("Find a place will occur our meeting")
        place_result = re.search(pattern_place, meeting_text)
        if place_result:
            place = place_result.group(1)
            logger.info(f"Place: {place}")
        else:
            logger.info("There is no meeting")

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


#ReadingFiles().regex_shorthand()
#ReadingFiles().regex_grouping()
ReadingFiles().regex_quantifiers()
