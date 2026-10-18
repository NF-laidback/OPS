import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Logging the run in default experiment

max_depth = 5
n_estimators = 60
mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth= max_depth, n_estimators= n_estimators)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_param("max_depth", max_depth)

    print("accuracy", accuracy)


