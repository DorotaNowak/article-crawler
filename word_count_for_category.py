import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt


dataset = pd.read_csv('articles.csv')
links = dataset['Link']

data = ""
for i in range(len(links)):
    cat = links[i].split('/')[3]
    if cat == 'sport':
        path = 'articles_txt/' + str(i) + '.txt'
        with open(path) as fp:
            file = fp.read()
        data += file
        data += "\n"

words = word_tokenize(data)

words = [w.lower() for w in words if w.isalnum()]

words_count = defaultdict(int)

for word in words:
    words_count[word] += 1


words_count_sorted = sorted(words_count.items(), key=lambda k_v: k_v[1], reverse=True)
x_val = [x[0] for x in words_count_sorted[:35]]
y_val = [x[1] for x in words_count_sorted[:35]]

plt.bar(x_val,y_val)
plt.xticks(rotation=45)
plt.xticks(fontsize=7)
plt.savefig('sport_top35')
plt.show()
