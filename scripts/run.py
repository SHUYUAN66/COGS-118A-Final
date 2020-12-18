
from sklearn.metrics import  make_scorer, f1_score, accuracy_score, mean_squared_error, average_precision_score, roc_auc_score, log_loss, recall_score, precision_score
from save import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler,  OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector as selector
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib


def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        if not CHECK_FOLDER:
            os.makedirs(i)
        else:
            return


def save_trails(pre_params_, preprocessor_, alg, scr, data, path=['all_models/', 'best_models/']):
    """
    alg: Algorithm, a dict(), each one is a name of that function {'knn':[KNN(),knn_params]}
    scr: scorings, a dict(), each one is a function such as 'acc':ACC 
    data: a dict() of dataset {'dataname':[trainset, testset]}
    path = ['all_models','best_models'] , to decide first chart or second chart.
    """
    # parameters
    params = pre_params_
    alg_name = list(alg)[0]
    data_name = list(data)[0]
    clsf = alg[alg_name][0]
    param = alg[alg_name][1]
    params.append(param)
    print(params)
    dataset = data[data_name][0]
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', clsf)])
    X = dataset.drop(columns=['target'])
    y = dataset.target
    record = {}
    for i in range(len(list(scr))):
        score_name = list(scr)[i]
        score = scr[score_name]
        trail_name = alg_name+data_name+score_name
        record[trail_name] = []
        details = {}
        # 4000 : 1000
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)
        clf = GridSearchCV(pipeline, params, scoring=score,
                           cv=5, n_jobs=-1, return_train_score=True)
        clf.fit(X_train, y_train)
        # For each trialwe use 4000 cases to train thedi erent models,1000 casesto calibrate the models and select the best parameters,and then report performance on the large final test set.
        save_models = os.path.join(path[0], alg_name, data_name, score_name)
        check_directory([save_models])
        joblib.dump(clf, os.path.join(
            save_models, 'all_models.pkl'))
        mean_train_grade = clf.cv_results_['mean_test_score']
        best_score = clf.best_score_
        best_params = clf.best_params_
        best_model = clf.best_estimator_
        details['mean_train_score'] = mean_train_grade
        details['best_score'] = best_score
        details['best_params'] = best_params
        save_best_model = os.path.join(
            path[1], alg_name, data_name, score_name)
        check_directory([save_best_model])
        details['best_estimator'] = best_model
        joblib.dump(best_model, os.path.join(
            save_best_model, 'best_model.pkl'))
        record[trail_name].append(details)
    return


if __name__ == '__main__':
    nsr_train = pd.read_csv('data/train/nsr.csv')
    nsr_test = pd.read_csv('data/test/nsr.csv')
    nsr_info = {'nursery': [nsr_train, nsr_test]}
    ordinal_transformer = Pipeline(
        steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
               ('encode', OrdinalEncoder())
               ])
    numeric_transformer = Pipeline(
        steps=[('miss', SimpleImputer()),  ('scaler', StandardScaler())
               ])
    knn_params = {'classifier': [KNeighborsClassifier()],
                  'classifier__n_neighbors': np.random.randint(1, 50, 10),
                  'classifier__weights': ['uniform', 'distance'],
                  'classifier__algorithm': ['brute']}
    knn_info = {'knn': [KNeighborsClassifier(), knn_params]}
    preprocessor_ordinal = ColumnTransformer(transformers=[('categoricals', ordinal_transformer,
                                                            selector(dtype_include=["object", "category"])),
                                                           ('numericals', numeric_transformer, selector(
                                                               dtype_include=["int", 'float']))
                                                           ])
    prep_org = [
        {'preprocessing__categoricals': [ordinal_transformer],
         'preprocessing__categoricals__miss__strategy': ['most_frequent']},
        {'preprocessing__numericals': [numeric_transformer],
         'preprocessing__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
    ]
    ACC = make_scorer(accuracy_score)
    PRC = make_scorer(precision_score, average='weighted')
    FSC = make_scorer(f1_score, average='macro')
    LFT = make_scorer(recall_score, average='macro')
    ROC = make_scorer(roc_auc_score)
    APR = make_scorer(average_precision_score, average='macro')
    RMS = make_scorer(mean_squared_error)
    MXE = make_scorer(log_loss)

    scores_info = {'ACC': ACC,
                   'PRC': PRC,
                   'FSC': FSC,
                   'LFT': LFT,
                   'ROC': ROC,
                   'APR': APR,
                   'RMS': RMS,
                   'MXE': MXE}
    save_trails(prep_org, preprocessor_ordinal,
                knn_info, scores_info, nsr_info)
