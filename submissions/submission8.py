import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# P(c) = numar aparitii sa fie dialect c / numar de inscrieri
# P(A, c) = de cate ori e dialect c cand e limba k / numarul de prop din limba k
# P(c, A) = P(A, c) * P(c) / P(A)

# vector boolean
# sum(log P(cuvant , c) + log P(c))
# P(cuvant, c) = frecventa cuvantului in dialectul c + 1 / frecventele totale (ale clasei) + dimensiunea vocabularului

# scikit learn (Multinomial Naive Bayes)

# language classifier is good!

# TODO: try KNN

def determine_language(phrase):
    phrase = vectorizer_all.transform(phrase)
    return mnb_language.predict(phrase)


def classify_naive_bayes(a1, a2, a3, p1, p2, p3, phrase):
    p_england = 0
    p_ireland = 0
    p_scotland = 0
    for word in phrase:
        p_england = p_england + (np.log10(((a1[1][np.where(a1[0] == word)] if len(np.where(a1[0] == word)) > 0 else 0) + 1) /(sum(a1[1]) + len(a1[0]))) + np.log10(p1))
        p_ireland = p_ireland + (np.log10(((a2[1][np.where(a2[0] == word)] if len(np.where(a2[0] == word)) > 0 else 0) + 1) / (sum(a2[1]) + len(a2[0]))) + np.log10(p2))
        p_scotland = p_scotland + (np.log10(((a3[1][np.where(a3[0] == word)] if len(np.where(a3[0] == word)) > 0 else 0) + 1) / (sum(a3[1]) + len(a3[0]))) + np.log10(p3))
    print(p_england, p_ireland, p_scotland)
    if max(p_england, p_ireland, p_scotland) == p_england:
        return 'England'
    elif max(p_england, p_ireland, p_scotland) == p_ireland:
        return 'Ireland'
    else:
        return 'Scotland'


data = pd.read_csv("train_data.csv").values

characters_to_remove = '\n\"\'-.,()[]{}¿?¡!+=«»:;'


for i in range(len(data)):
    for c in characters_to_remove:
        data[i][1] = data[i][1].replace(c, " ")
    data[i][1] = np.array(data[i][1].split())

# for splitting

# split = int(len(data) / 10)
# random.shuffle(data)
# test_data = data[:split]
# train_data = data[split:]

# [0] BOW, [1] training_data, [2] training_labels

BOW_ALL = [[], [], []]

BOW_Dansk = [[], [], []]
BOW_Deutsch = [[], [], []]
BOW_Espanol = [[], [], []]
BOW_Italiano = [[], [], []]
BOW_Nederlands = [[], [], []]


for case in data:  # change to data for submission
    BOW_ALL[0].extend(case[1])
    BOW_ALL[1].append(case[1])
    BOW_ALL[2].append(case[0])
    if case[0] == 'dansk':
        BOW_Dansk[0].extend(case[1])
        BOW_Dansk[1].append(case[1])
        BOW_Dansk[2].append(case[2])
    elif case[0] == 'Deutsch':
        BOW_Deutsch[0].extend(case[1])
        BOW_Deutsch[1].append(case[1])
        BOW_Deutsch[2].append(case[2])
    elif case[0] == 'español':
        BOW_Espanol[0].extend(case[1])
        BOW_Espanol[1].append(case[1])
        BOW_Espanol[2].append(case[2])
    elif case[0] == 'italiano':
        BOW_Italiano[0].extend(case[1])
        BOW_Italiano[1].append(case[1])
        BOW_Italiano[2].append(case[2])
    elif case[0] == 'Nederlands':
        BOW_Nederlands[0].extend(case[1])
        BOW_Nederlands[1].append(case[1])
        BOW_Nederlands[2].append(case[2])


vectorizer_all = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 1000)
vectorizer_all.fit(BOW_ALL[1])
BOW_ALL[1] = vectorizer_all.transform(BOW_ALL[1])
mnb_language = MultinomialNB(alpha=0.5)
mnb_language.fit(BOW_ALL[1], BOW_ALL[2])

mnb_Dansk = KNeighborsClassifier(n_neighbors=5)
vectorizer_Dansk = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 100000)
vectorizer_Dansk.fit(BOW_Dansk[1])
X_train_Dansk = vectorizer_Dansk.transform(BOW_Dansk[1])
mnb_Dansk.fit(X_train_Dansk, BOW_Dansk[2])

mnb_Deutsch = KNeighborsClassifier(n_neighbors=5)
vectorizer_Deutsch = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 100000)
vectorizer_Deutsch.fit(BOW_Deutsch[1])
X_train_Deutsch = vectorizer_Deutsch.transform(BOW_Deutsch[1])
mnb_Deutsch.fit(X_train_Deutsch, BOW_Deutsch[2])

mnb_Espanol = KNeighborsClassifier(n_neighbors=5)
vectorizer_Espanol = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 100000)
vectorizer_Espanol.fit(BOW_Espanol[1])
X_train_Espanol = vectorizer_Espanol.transform(BOW_Espanol[1])
mnb_Espanol.fit(X_train_Espanol, BOW_Espanol[2])

mnb_Italiano = KNeighborsClassifier(n_neighbors=5)
vectorizer_Italiano = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 100000)
vectorizer_Italiano.fit(BOW_Italiano[1])
X_train_Italiano = vectorizer_Italiano.transform(BOW_Italiano[1])
mnb_Italiano.fit(X_train_Italiano, BOW_Italiano[2])

mnb_Nederlands = KNeighborsClassifier(n_neighbors=5)
vectorizer_Nederlands = CountVectorizer(tokenizer = lambda x:x,    # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor = lambda x:x,  #  data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features = 100000)
vectorizer_Nederlands.fit(BOW_Nederlands[1])
X_train_Nederlands = vectorizer_Nederlands.transform(BOW_Nederlands[1])
mnb_Nederlands.fit(X_train_Nederlands, BOW_Nederlands[2])

predictions = []

# for splitting

# test_labels = [label[2] for label in test_data]
# test_data = [test[1] for test in test_data]

data = pd.read_csv("test_data.csv").values

test_data = []

for i in range(len(data)):
    for c in characters_to_remove:
        data[i][0] = data[i][0].replace(c, " ")
    data[i][0] = np.array(data[i][0].split())
    test_data.append(data[i][0])

file = open('submission.csv', 'w', newline='')
writer = csv.writer(file)

for i in tqdm(range(len(test_data))):
    language = determine_language([test_data[i]])
    if language == 'dansk':
        X_test = vectorizer_Dansk.transform([test_data[i]])
        predictions.extend(mnb_Dansk.predict(X_test))
    elif language == 'Deutsch':
        X_test = vectorizer_Deutsch.transform([test_data[i]])
        predictions.extend(mnb_Deutsch.predict(X_test))
    elif language == 'español':
        X_test = vectorizer_Espanol.transform([test_data[i]])
        predictions.extend(mnb_Espanol.predict(X_test))
    elif language == 'italiano':
        X_test = vectorizer_Italiano.transform([test_data[i]])
        predictions.extend(mnb_Italiano.predict(X_test))
    elif language == 'Nederlands':
        X_test = vectorizer_Nederlands.transform([test_data[i]])
        predictions.extend(mnb_Nederlands.predict(X_test))

# result = np.bincount(np.array(predictions) == test_labels)
# print(result)

writer.writerow(['id', 'label'])
for i in range(len(predictions)):
    writer.writerow([(i + 1).__str__(), predictions[i]])

file.close()
