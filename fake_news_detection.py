# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:39:52 2021

@author: doguilmak

"""
#%%
# 1. Importing Libraries

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
df=pd.read_csv('news.csv')
LABEL=df['label']

# 2.2. Get shape and head
print('\n', LABEL.value_counts(), '\n')
print(df.shape)
print(df.head(10))

# 2.3. DataFlair - Get the labels
labels=df.label
print(labels.head(10))

# 2.3. Plot Faiure Types on Histogram
plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')    
sns.histplot(data=LABEL)
plt.title("FAKE-REAL on Histogram")
plt.xlabel("News Types")
plt.ylabel("News in Total")
#plt.savefig('Plots/hist_failure_types')
plt.show()

# 2.4. Train - Test Split
x_train, x_test, y_train, y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# 2.5. Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# 2.6. Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# 2.7. Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 2.8. Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy score: {score} %')

# 2.9. Creating Confusion Matrix
print(confusion_matrix(y_test, y_pred, labels=['FAKE','REAL']))

#%%
# 4 XGBoost

# 4.1 Importing libraries
from xgboost import XGBClassifier

classifier= XGBClassifier()
classifier.fit(tfidf_train, y_train)

y_pred = classifier.predict(tfidf_test)

# 4.2. Building confusion matrix
cm2 = confusion_matrix(y_pred, y_test)  #  Comparing results
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 4.3. Accuracy score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
