import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# P(c) = numar aparitii sa fie dialect c / numar de inscrieri
# P(A, c) = de cate ori e dialect c cand e limba k / numarul de prop din limba k
# P(c, A) = P(A, c) * P(c) / P(A)

# vector boolean
# sum(log P(cuvant , c) + log P(c))
# P(cuvant, c) = frecventa cuvantului in dialectul c + 1 / frecventele totale (ale clasei) + dimensiunea vocabularului

# scikit learn (Multinomial Naive Bayes)


data = pd.read_csv("train_data.csv").values

characters_to_remove = '\n\"\'-.,()[]{}¿?¡!+=«»:;'

train_data = []
train_labels = []

for i in range(len(data)):
    for c in characters_to_remove:
        data[i][1] = data[i][1].replace(c, " ")
    data[i][1] = np.array(data[i][1].split())
    train_data.append(data[i][1])
    train_labels.append(data[i][2])


predictions = []

data = pd.read_csv("test_data.csv").values

test_data = []

for i in range(len(data)):
    for c in characters_to_remove:
        data[i][0] = data[i][0].replace(c, " ")
    data[i][0] = np.array(data[i][0].split())
    test_data.append(data[i][0])


file = open('submission.csv', 'w', newline='')
writer = csv.writer(file)

mnb = MultinomialNB(alpha=0.5)


train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

vectorizer = CountVectorizer(tokenizer = lambda x:x,
                             preprocessor = lambda x:x,
                             max_features = 1000)
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)
X_test = vectorizer.transform(test_data)

mnb.fit(X_train, train_labels)

predictions = mnb.predict(X_test)
writer.writerow(['id', 'label'])

for i in range(len(predictions)):
    writer.writerow([(i + 1).__str__(), predictions[i]])

file.close()
