import requests
from bs4 import BeautifulSoup
import pandas as pd


def save_to_file(id, text):
    path = 'articles_txt/' + str(id) + '.txt'
    with open(path, 'w+', encoding='cp1250') as file:
        file.write(text)
    file.close()


def process_link(id, article_link, text=''):
    res = requests.get(article_link)
    soup = BeautifulSoup(res.text, features='html.parser')
    header = soup.find('h1', class_='_1RXXoWo4WDx3KkMPFZutNI')
    text += f'{header.text} '
    subtitle = soup.find('p', class_='_3eutbWyfJM8V9XMMgNVJMY')
    text += f'{subtitle.text} '
    for a in soup.findAll('a', href=True):
        a.extract()
    content = soup.findAll(True, {'class': ['UfY35hb-RNY0K7QeuQbd', '_2Oh2J73ZaVkn5q5Z193TJG']})
    for i in range(len(content)):
        text += content[i].text
        if len(content[i].text) > 0 and content[i].text != ' ':
            text += ' '
    save_to_file(id, text)


df = pd.read_csv('articles.csv')

r = requests.get('https://www.onet.pl/')
soup = BeautifulSoup(r.text, features="html.parser")
articles_tags = soup.find_all(attrs={"data-art-type": "article"})
for a in articles_tags:
    link = a.attrs["href"]
    if 'https://www.onet.pl/' in link:
        if not df['Link'].str.contains(link).any():
            max_id = df.shape[0]
            try:
                process_link(max_id, link)
            except:
                continue
            df = pd.concat([df, pd.DataFrame({'Id': [max_id], 'Link': [link]})], ignore_index=True)
df.to_csv('articles.csv', index=False)
