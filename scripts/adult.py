# Basic
from sklearn.metrics import  make_scorer, f1_score, accuracy_score, mean_squared_error, precision_recall_curve, confusion_matrix, classification_report, precision_score, average_precision_score, matthews_corrcoef, roc_curve, precision_recall_fscore_support, log_loss,recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pprintpp import pprint
import timeit

# preprocessing
# StandardScaler(), KFold()
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
# selecting model
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve, learning_curve
# Algorithms
from supervised.automl import AutoML
from sklearn.utils.fixes import loguniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion, Pipeline




for score in scorings:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(
        pipeline, params, scoring=score, cv=5,
        n_jobs=-1, refit=callable)
    clf.fit(X_train, y_train)
    print("**pipeline**:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(params)
    print("Best parameters set found on development set:")
    print(" ", clf.best_params_)
    print("Best estimatpr found:")
    print(" ", clf.best_estimator_)
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
ad = pd.read_csv('notebooks/cleaning_data/adult.csv')
