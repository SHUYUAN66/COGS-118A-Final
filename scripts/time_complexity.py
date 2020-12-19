# TODO: havent done anything yet! 
import sys
import os
sys.path.insert(
    0, '../packages')
from time_complexity_evaluator import ComplexityEvaluator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression


def random_data_regression(n, p):
    return np.random.rand(n, p), np.random.rand(n)


def random_data_classification(n, p):
    return np.random.rand(n, p), np.random.binomial(1, 0.5, n)


regression_models = [RandomForestRegressor(),
                     ExtraTreesRegressor(),
                     AdaBoostRegressor(),
                     LinearRegression(),
                     SVR()]

classification_models = [RandomForestClassifier(),
                         ExtraTreesClassifier(),
                         AdaBoostClassifier(),
                         SVC(),
                         LogisticRegression(),
                         LogisticRegression(solver='sag')]

names = ["RandomForestRegressor",
         "ExtraTreesRegressor",
         "AdaBoostRegressor",
         "LinearRegression",
         "SVR",
         "RandomForestClassifier",
         "ExtraTreesClassifier",
         "AdaBoostClassifier",
         "SVC",
         "LogisticRegression(solver=liblinear)",
         "LogisticRegression(solver=sag)"]

complexity_evaluator = ComplexityEvaluator(
    [500, 1000, 2000, 5000],
    [5, 10, 20, 50])

i = 0
for model in regression_models:
    res = complexity_evaluator.Run(model, random_data_regression)[0]
    print(names[i] + ' | ' + str(round(res[0], 2)) +
          ' | ' + str(round(res[1], 2)))
    i = i + 1

for model in classification_models:
    res = complexity_evaluator.Run(model, random_data_classification)[0]
    print(names[i] + ' | ' + str(round(res[0], 2)) +
          ' | ' + str(round(res[1], 2)))
    i = i + 1
