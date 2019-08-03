import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train_df=pd.read_csv("train.csv")
train_df.head()

test_df=pd.read_csv("test.csv")
test_df.head()

temp_df=pd.DataFrame()
temp_df["PassengerId"]=test_df["PassengerId"]
temp_df.head()

train_df=train_df.drop(["Ticket", "Cabin", "Name", "PassengerId", "Embarked"], axis=1).fillna(-99999)
train_df.head()

test_df=test_df.drop(["Ticket", "Cabin", "Name", "PassengerId", "Embarked"], axis=1).fillna(-99999)
test_df.head()

train_df["Sex"]=train_df["Sex"].replace("male", 1)
train_df["Sex"]=train_df["Sex"].replace("female", 0)
train_df.head()

test_df["Sex"]=test_df["Sex"].replace("male", 1)
test_df["Sex"]=test_df["Sex"].replace("female", 1)
test_df.head()

x_train=np.array(train_df.drop(["Survived"], 1))
y_train=np.array(train_df["Survived"])
x_test=np.array(test_df)

logreg=LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_test)
accuracy= round(logreg.score(x_train, y_train)*100, 2)

submit_df=pd.DataFrame(np.reshape(y_pred, (len(y_pred))), columns=["Survived"])
submit_df["PassengerId"]=temp_df["PassengerId"]
submit_df.head()

submit_df.to_csv("titanic01.csv")
submit_df