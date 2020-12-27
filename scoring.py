#%% Import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
os.chdir(os.path.dirname(__file__))

data = pd.read_csv("./data/data_avec_etiquettes.txt", sep="\t")

encoder = OrdinalEncoder()
encoder.fit(data[["V160", "V161", "V162"]])
data[["V160", "V161", "V162"]] = encoder.transform(data[["V160", "V161", "V162"]])

X = data.iloc[:,0:199]
y = data.iloc[:,199]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=250000, random_state=42)
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
taille = np.arange(start=1, stop=250001, step=1)
taille = taille / n

plt.title('Courbe de gain')
plt.xlabel('Taille de cible')
plt.ylabel('Rappel')

plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter(taille, taille, marker='.', color='blue')
plt.scatter(taille, rappel, marker='.', color='red')

plt.show()
