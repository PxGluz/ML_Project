import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# P(c) = numar aparitii sa fie dialect c / numar de inscrieri
# P(A, c) = de cate ori e dialect c cand e limba k / numarul de prop din limba k
# P(c, A) = P(A, c) * P(c) / P(A)

# vector boolean
# sum(log P(cuvant , c) + log P(c))
# P(cuvant, c) = frecventa cuvantului in dialectul c + 1 / frecventele totale (ale clasei) + dimensiunea vocabularului

# scikit learn (Multinomial Naive Bayes)


def determine_language(phrase):
    language = [0, 0, 0, 0, 0]
    for word in phrase:
        language[0] += (1 if word in BOW_Dansk[0] else 0)
        language[1] += (1 if word in BOW_Deutsch[0] else 0)
        language[2] += (1 if word in BOW_Espanol[0] else 0)
        language[3] += (1 if word in BOW_Italiano[0] else 0)
        language[4] += (1 if word in BOW_Nederlands[0] else 0)
    # print(language)
    result = np.argmax(language)
    if result == 0:
        # print("got dansk")
        return 'dansk'
    if result == 1:
        # print("got Deutsch")
        return 'Deutsch'
    if result == 2:
        # print("got español")
        return 'español'
    if result == 3:
        # print("got italiano")
        return 'italiano'
    if result == 4:
        # print("got Nederlands")
        return 'Nederlands'


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
    data[i][1] = data[i][1].split()

train_data = data

BOW_Dansk = [[], []]
BOW_Deutsch = [[], []]
BOW_Espanol = [[], []]
BOW_Italiano = [[], []]
BOW_Nederlands = [[], []]

TOT_Dansk = 0.
TOT_Deutsch = 0.
TOT_Espanol = 0.
TOT_Italiano = 0.
TOT_Nederlands = 0.

BOW_Dansk_England = [[], []]
BOW_Dansk_Ireland = [[], []]
BOW_Dansk_Scotland = [[], []]

BOW_Deutsch_England = [[], []]
BOW_Deutsch_Ireland = [[], []]
BOW_Deutsch_Scotland = [[], []]

BOW_Espanol_England = [[], []]
BOW_Espanol_Ireland = [[], []]
BOW_Espanol_Scotland = [[], []]

BOW_Italiano_England = [[], []]
BOW_Italiano_Ireland = [[], []]
BOW_Italiano_Scotland = [[], []]

BOW_Nederlands_England = [[], []]
BOW_Nederlands_Ireland = [[], []]
BOW_Nederlands_Scotland = [[], []]

P_Dansk_England = 0.
P_Dansk_Ireland = 0.
P_Dansk_Scotland = 0.

P_Deutsch_England = 0.
P_Deutsch_Ireland = 0.
P_Deutsch_Scotland = 0.

P_Espanol_England = 0.
P_Espanol_Ireland = 0.
P_Espanol_Scotland = 0.

P_Italiano_England = 0.
P_Italiano_Ireland = 0.
P_Italiano_Scotland = 0.

P_Nederlands_England = 0.
P_Nederlands_Ireland = 0.
P_Nederlands_Scotland = 0.

for case in train_data:
    case[1] = list(set(case[1]))
    if case[0] == 'dansk':
        TOT_Dansk += 1
        BOW_Dansk[0].extend(case[1])
        if case[2] == 'England':
            BOW_Dansk_England[0].extend(case[1])
            P_Dansk_England += 1
        elif case[2] == 'Scotland':
            BOW_Dansk_Scotland[0].extend(case[1])
            P_Dansk_Scotland += 1
        elif case[2] == 'Ireland':
            BOW_Dansk_Ireland[0].extend(case[1])
            P_Dansk_Ireland += 1
    elif case[0] == 'Deutsch':
        BOW_Deutsch[0].extend(case[1])
        TOT_Deutsch += 1
        if case[2] == 'England':
            BOW_Deutsch_England[0].extend(case[1])
            P_Deutsch_England += 1
        elif case[2] == 'Scotland':
            BOW_Deutsch_Scotland[0].extend(case[1])
            P_Deutsch_Scotland += 1
        elif case[2] == 'Ireland':
            BOW_Deutsch_Ireland[0].extend(case[1])
            P_Deutsch_Ireland += 1
    elif case[0] == 'español':
        BOW_Espanol[0].extend(case[1])
        TOT_Espanol += 1
        if case[2] == 'England':
            BOW_Espanol_England[0].extend(case[1])
            P_Espanol_England += 1
        elif case[2] == 'Scotland':
            BOW_Espanol_Scotland[0].extend(case[1])
            P_Espanol_Scotland += 1
        elif case[2] == 'Ireland':
            BOW_Espanol_Ireland[0].extend(case[1])
            P_Espanol_Ireland += 1
    elif case[0] == 'italiano':
        BOW_Italiano[0].extend(case[1])
        TOT_Italiano += 1
        if case[2] == 'England':
            BOW_Italiano_England[0].extend(case[1])
            P_Italiano_England += 1
        elif case[2] == 'Scotland':
            BOW_Italiano_Scotland[0].extend(case[1])
            P_Italiano_Scotland += 1
        elif case[2] == 'Ireland':
            BOW_Italiano_Ireland[0].extend(case[1])
            P_Italiano_Ireland += 1
    elif case[0] == 'Nederlands':
        BOW_Nederlands[0].extend(case[1])
        TOT_Nederlands += 1
        if case[2] == 'England':
            BOW_Nederlands_England[0].extend(case[1])
            P_Nederlands_England += 1
        elif case[2] == 'Scotland':
            BOW_Nederlands_Scotland[0].extend(case[1])
            P_Nederlands_Scotland += 1
        elif case[2] == 'Ireland':
            BOW_Nederlands_Ireland[0].extend(case[1])
            P_Nederlands_Ireland += 1

BOW_Dansk[0], BOW_Dansk[1] = np.unique(BOW_Dansk[0], return_counts=True)
BOW_Deutsch[0], BOW_Deutsch[1] = np.unique(BOW_Deutsch[0], return_counts=True)
BOW_Espanol[0], BOW_Espanol[1] = np.unique(BOW_Espanol[0], return_counts=True)
BOW_Italiano[0], BOW_Italiano[1] = np.unique(BOW_Italiano[0], return_counts=True)
BOW_Nederlands[0], BOW_Nederlands[1] = np.unique(BOW_Nederlands[0], return_counts=True)

BOW_Dansk_England[0], BOW_Dansk_England[1] = np.unique(BOW_Dansk_England[0], return_counts=True)
BOW_Dansk_Ireland[0], BOW_Dansk_Ireland[1] = np.unique(BOW_Dansk_Ireland[0], return_counts=True)
BOW_Dansk_Scotland[0], BOW_Dansk_Scotland[1] = np.unique(BOW_Dansk_Scotland[0], return_counts=True)
P_Dansk_England /= TOT_Dansk
P_Dansk_Ireland /= TOT_Dansk
P_Dansk_Scotland /= TOT_Dansk

BOW_Deutsch_England[0], BOW_Deutsch_England[1] = np.unique(BOW_Deutsch_England[0], return_counts=True)
BOW_Deutsch_Ireland[0], BOW_Deutsch_Ireland[1] = np.unique(BOW_Deutsch_Ireland[0], return_counts=True)
BOW_Deutsch_Scotland[0], BOW_Deutsch_Scotland[1] = np.unique(BOW_Deutsch_Scotland[0], return_counts=True)
P_Deutsch_England /= TOT_Deutsch
P_Deutsch_Ireland /= TOT_Deutsch
P_Deutsch_Scotland /= TOT_Deutsch

BOW_Espanol_England[0], BOW_Espanol_England[1] = np.unique(BOW_Espanol_England[0], return_counts=True)
BOW_Espanol_Ireland[0], BOW_Espanol_Ireland[1] = np.unique(BOW_Espanol_Ireland[0], return_counts=True)
BOW_Espanol_Scotland[0], BOW_Espanol_Scotland[1] = np.unique(BOW_Espanol_Scotland[0], return_counts=True)
P_Espanol_England /= TOT_Espanol
P_Espanol_Ireland /= TOT_Espanol
P_Espanol_Scotland /= TOT_Espanol

BOW_Italiano_England[0], BOW_Italiano_England[1] = np.unique(BOW_Italiano_England[0], return_counts=True)
BOW_Italiano_Ireland[0], BOW_Italiano_Ireland[1] = np.unique(BOW_Italiano_Ireland[0], return_counts=True)
BOW_Italiano_Scotland[0], BOW_Italiano_Scotland[1] = np.unique(BOW_Italiano_Scotland[0], return_counts=True)
P_Italiano_England /= TOT_Italiano
P_Italiano_Ireland /= TOT_Italiano
P_Italiano_Scotland /= TOT_Italiano

BOW_Nederlands_England[0], BOW_Nederlands_England[1] = np.unique(BOW_Nederlands_England[0], return_counts=True)
BOW_Nederlands_Ireland[0], BOW_Nederlands_Ireland[1] = np.unique(BOW_Nederlands_Ireland[0], return_counts=True)
BOW_Nederlands_Scotland[0], BOW_Nederlands_Scotland[1] = np.unique(BOW_Nederlands_Scotland[0], return_counts=True)
P_Nederlands_England /= TOT_Nederlands
P_Nederlands_Ireland /= TOT_Nederlands
P_Nederlands_Scotland /= TOT_Nederlands

predictions = []

data = pd.read_csv("test_data.csv").values

for i in range(len(data)):
    for c in characters_to_remove:
        data[i][0] = data[i][0].replace(c, " ")
    data[i][0] = data[i][0].split()

test_data = data

file = open('submission.csv', 'w')
writer = csv.writer(file)

for i in range(len(test_data)):
    print(i.__str__() + '/' + len(test_data).__str__())
    predict = ""
    lang = determine_language(test_data[i][0])

    if lang == 'dansk':
        predict = classify_naive_bayes(BOW_Dansk_England, BOW_Dansk_Ireland, BOW_Dansk_Scotland, P_Dansk_England, P_Dansk_Ireland, P_Dansk_Scotland, test_data[i][0])
    elif lang == 'Deutsch':
        predict = classify_naive_bayes(BOW_Deutsch_England, BOW_Deutsch_Ireland, BOW_Deutsch_Scotland, P_Deutsch_England, P_Deutsch_Ireland, P_Deutsch_Scotland, test_data[i][0])
    elif lang == 'español':
        predict = classify_naive_bayes(BOW_Espanol_England, BOW_Espanol_Ireland, BOW_Espanol_Scotland, P_Espanol_England, P_Espanol_Ireland, P_Espanol_Scotland, test_data[i][0])
    elif lang == 'italiano':
        predict = classify_naive_bayes(BOW_Italiano_England, BOW_Italiano_Ireland, BOW_Italiano_Scotland, P_Italiano_England, P_Italiano_Ireland, P_Italiano_Scotland, test_data[i][0])
    elif lang == 'Nederlands':
        predict = classify_naive_bayes(BOW_Nederlands_England, BOW_Nederlands_Ireland, BOW_Nederlands_Scotland, P_Nederlands_England, P_Nederlands_Ireland, P_Nederlands_Scotland, test_data[i][0])
    writer.writerow([(i+1).__str__(), predict])

file.close()
