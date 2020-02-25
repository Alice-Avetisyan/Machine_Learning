from googlesearch import search
import requests
from bs4 import BeautifulSoup

print("What is your query?")
while True:
    user_query = input()
    url_links = list(search(user_query, tld="com", num=10, stop=3, pause=1))
    #print(url_links)

    page_source = requests.get(url_links[0])

    soup = BeautifulSoup(page_source.text, 'html.parser')
    print(soup.title.text)
    print(soup.p.text)
    print(soup.find_all('p'))

    print("Would you like to continue?")
    user_inp = input()
    if user_inp == 'Yes':
        print("What would you like to know?")
    else:
        break

