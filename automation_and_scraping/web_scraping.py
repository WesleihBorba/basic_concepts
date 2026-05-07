# Goal: Getting data on internet
import requests
from bs4 import BeautifulSoup


class WebScraping:

    def __init__(self):
        self.stock = 'AAPL'
        self.link = f'https://finance.yahoo.com/quote/{self.stock}/'
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def getting_information(self):
        response = requests.get(self.link, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Price of Stock
        price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        print("Actual Price:", price_tag.text if price_tag else "Not Find")

        market_cap = soup.find("fin-streamer", {"data-field": "marketCap"})
        print("Market Cap:", market_cap.text if market_cap else "Not Find")


class_scraping = WebScraping()
class_scraping.getting_information()