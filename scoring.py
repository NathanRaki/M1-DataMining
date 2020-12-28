#%% Import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
os.chdir(os.path.dirname(__file__))

startTime = datetime.now()

cols = ['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V197', 'V198', 'V199', 'V200']
data = pd.read_csv("./data/data_avec_etiquettes.txt", sep="\t", usecols=cols)

encoder = OrdinalEncoder()
encoder.fit(data[["V160", "V161", "V162"]])
data[["V160", "V161", "V162"]] = encoder.transform(data[["V160", "V161", "V162"]])

X = data.iloc[:,0:41]
y = data.iloc[:,41]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
y_test = y_test.reset_index()
y_test = y_test.drop(labels='index', axis=1)

lr = LogisticRegression(solver='liblinear')
modele_all = lr.fit(X_train,y_train)
probas = lr.predict_proba(X_test)

print(lr.classes_)

score = probas[:,7]
pos = pd.get_dummies(y_test)
pos = pos['V200_m16']

print(pos)

npos = np.sum(pos)
index = np.argsort(score)
index = index[::-1]
sort_pos = pos[index]
print(sort_pos)
cpos = np.cumsum(sort_pos)
rappel = cpos/npos
n = y_test.shape[0]
taille = np.arange(start=1, stop=148208, step=1)
taille = taille / n

plt.title('Courbe de gain')
plt.xlabel('Taille de cible')
plt.ylabel('Rappel')

plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter(taille, taille, marker='.', color='blue')
plt.scatter(taille, rappel, marker='.', color='red')

plt.show()
print('Temps d\'exécution: ', datetime.now() - startTime)
