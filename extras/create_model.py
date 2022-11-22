import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



# New imports




import streamlit as st

import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()




# Import the trees from sklearn
from sklearn import tree

# Helper function to split our data



# Helper fuctions to evaluate our model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

# Helper function for hyper-parameter turning.
from sklearn.model_selection import GridSearchCV

# Import our Decision Tree
from sklearn.tree import DecisionTreeClassifier 


# Import our Random Forest 
from sklearn.ensemble import RandomForestClassifier


# Library for visualizing our tree
# If you get an error, run 'conda install python-graphviz' in your terminal (without the quotes).
import graphviz

#


df = pd.read_csv('../data/titanic.csv')

df = pd.get_dummies(df, columns=['sex', 'pclass'], drop_first=True)

selected_features = ['fare', 'pclass_2', 'pclass_3', 'sex_male']

X = df[selected_features]

y = df['survived']


model = LogisticRegression()

model.fit(X, y)


pickle.dump(model, open('models/model.pkl', 'wb'))


# loading the ESRB model



df = pd.read_csv("video_games_esrb_rating.csv")


df["strong_language"] = df["strong_janguage"]

df.drop("strong_janguage", axis=1, inplace=True)
# dropping the wrongly spelled one

df.drop("console", axis=1, inplace=True)

# dropping console because it is not a useful feature

column_names = list(df.columns)



# no nulls



# we have 33 duplicate rows that we need to remove



df = df.drop_duplicates()





df["num_descriptors"] = 999

# just making a placeholder

list_descriptors = list(df.columns)
list_descriptors.remove("title")

list_descriptors.remove("esrb_rating")
list_descriptors.remove("no_descriptors")
list_descriptors.remove("num_descriptors")

df["num_descriptors"] = df[list_descriptors].sum(axis=1)



df["no_descriptors"]= np.where((df["num_descriptors"] == 0), 1, 0)


encode = {'E' : 0,
          'ET': 1,
          'T' : 2,
          'M' : 3}

df["esrb_encoded"] = df["esrb_rating"].map(encode)






selected_features = list(df.columns)



selected_features.remove("title")
selected_features.remove("esrb_rating")
selected_features.remove("esrb_encoded")

selected_features.remove("no_descriptors")
# removing this as
# num_descriptor is a better version of this
# and makes user input easier



X = df[selected_features]

y = df["esrb_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


final_model = RandomForestClassifier()

X = df[selected_features]

y = df["esrb_rating"]




final_model.fit(X, y)

y_pred = final_model.predict(X_test)



y_pred = final_model.predict(X)
    

# save esrb model

pickle.dump(final_model, open('models/final_model.pkl', 'wb'))

# maybe save the cleaned data in a csv so we do not have to keep cleaning
# and redoing certain parts

