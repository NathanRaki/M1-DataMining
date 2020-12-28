#%% Preparing Data
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
os.chdir(os.path.dirname(__file__))

cols = ['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V197', 'V198', 'V199', 'V200']
#cols = ['V14', 'V84', 'V96', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V166', 'V168', 'V170', 'V171', 'V175', 'V180', 'V181', 'V184', 'V185', 'V189', 'V190', 'V193', 'V200']
data = pd.read_csv("./data/data_avec_etiquettes.txt", sep="\t", usecols=cols)

#%% Test for LinearRegression
from sklearn.linear_model import LinearRegression

startTime = datetime.now()

encoder = OrdinalEncoder()
encoder.fit(data[["V160", "V161", "V162", "V200"]])
data[["V160", "V161", "V162", "V200"]] = encoder.transform(data[["V160", "V161", "V162", "V200"]])

X = data.iloc[:,0:41]
y = data.iloc[:,41]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=250000, random_state=42)
y_test = y_test.reset_index()
y_test = y_test.drop(labels='index', axis=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))
print('Temps d\'exécution: ', datetime.now() - startTime)

#%% Test for Logistic Regression
from sklearn.linear_model import LogisticRegression

startTime = datetime.now()

encoder = OrdinalEncoder()
encoder.fit(data[["V160", "V161", "V162"]])
data[["V160", "V161", "V162"]] = encoder.transform(data[["V160", "V161", "V162"]])

X = data.iloc[:,0:41]
y = data.iloc[:,41]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
y_test = y_test.reset_index()
y_test = y_test.drop(labels='index', axis=1)

regressor = LogisticRegression(solver='liblinear')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Train Score: ', regressor.score(X_train, y_train))
print('Test Score: ', regressor.score(X_test, y_test))
print('Temps d\'exécution: ', datetime.now() - startTime)