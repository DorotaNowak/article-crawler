import pandas as pd

dataset = pd.read_csv('articles.csv')
links = dataset['Link']

categories = dict()

for i in range(len(links)):
    cat = links[i].split('/')[3]
    if cat in categories:
        categories[cat] += 1
    else:
        categories[cat] = 1

print(categories)