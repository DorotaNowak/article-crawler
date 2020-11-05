import lemmatizer
import pandas as pd
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt

dataset = pd.read_csv('articles.csv')
links = dataset['Link']

# IDs of articles from each category

indices = {"informacje" : [], "sport" : [], "styl-zycia" : [], "turystyka" : [],
           "kultura" : [], "film" : [], "biznes" : [], "muzyka" : []}

for i in range(len(links)):
    cat = links[i].split('/')[3]
    indices[cat].append(i)


info_len = int(len(indices["informacje"])*0.8)
info_train = indices["informacje"][:info_len]
info_test = indices["informacje"][info_len:]

rest = ""
rest_train = []
rest_test = []

for key, value in indices.items():
    if key != "informacje":
        cat_len = int(len(value)*0.9)
        cat_train = value[:cat_len]
        rest_train += cat_train
        cat_test = value[cat_len:]
        rest_test += cat_test

info_train_data = ""
rest_train_data = ""

for i in info_train:
    path = 'articles_txt/' + str(i) + '.txt'
    with open(path) as fp:
        file = fp.read()
    info_train_data += file
    info_train_data += "\n"

for i in rest_train:
    path = 'articles_txt/' + str(i) + '.txt'
    with open(path) as fp:
        file = fp.read()
    rest_train_data += file
    rest_train_data += "\n"

def process(data):
    words = word_tokenize(data)
    words = [w.lower() for w in words if w.isalnum()]
    words = [lemmatizer.remove_diminutive(word) for word in words]
    words = [lemmatizer.remove_adjective_ends(word) for word in words]
    words = [lemmatizer.remove_adverbs_ends(word) for word in words]
    words = [lemmatizer.remove_general_ends(word) for word in words]
    words = [lemmatizer.remove_plural_forms(word) for word in words]
    words = [lemmatizer.remove_nouns(word) for word in words]
    words = [lemmatizer.remove_verbs_ends(word) for word in words]
    return words

info_words = process(info_train_data)
rest_words = process(rest_train_data)

words_count_info = defaultdict(int)
words_count_rest = defaultdict(int)

for word in info_words:
    words_count_info[word] += 1

for word in rest_words:
    words_count_rest[word] += 1

alpha = 0.01

I = set(info_words)
L = set(rest_words)

I.update(L)
W = len(I) # number of all diticnt words in harry_potter and lord_of_rings


def predict(sent):
    words = process(sent)

    def calc_info(sent):
        result = np.log(0.5)

        for word in words:
            result += np.log((words_count_info[word] + alpha) / (len(info_words) + W * alpha))

        return result

    def calc_rest(sent):
        result = np.log(0.5)

        for word in words:
            result += np.log((words_count_rest[word] + alpha) / (len(rest_words) + W * alpha))

        return result

    return 1 if calc_info(sent) > calc_rest(sent) else 0

correct = 0
for i in info_test:
    path = 'articles_txt/' + str(i) + '.txt'
    with open(path) as fp:
        file = fp.read()
        pred = predict(file)
        if pred == 1:
            correct +=1

for i in rest_test:
    path = 'articles_txt/' + str(i) + '.txt'
    with open(path) as fp:
        file = fp.read()
        pred = predict(file)
        if pred == 0:
            correct +=1

print(correct/(len(info_test)+len(rest_test)))

print(predict("Podróże do Azji wstrzymane."))