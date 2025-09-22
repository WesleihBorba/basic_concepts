# Goal: Getting data on internet
import requests
from bs4 import BeautifulSoup


class WebScraping:

    def __init__(self):
        self.link = None




webpage_response = requests.get('https://content.codecademy.com/courses/beautifulsoup/shellter.html')

webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

print(soup.p)
print(soup.p.string)

for child in soup.div.children:
    print(child)

for parent in soup.li.parents:
    print(parent)

    