
import numpy as np
from supervised.automl import AutoML
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from pprintpp import pprint
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
import sys
sys.path.insert(0, '/scripts')
#from pipelines import *
#from scores import *
from pipe_score import *

# Basic
from sklearn.metrics import  make_scorer, f1_score, accuracy_score, mean_squared_error, precision_recall_curve, confusion_matrix, classification_report, precision_score, average_precision_score, matthews_corrcoef, roc_curve, precision_recall_fscore_support, log_loss, recall_score

# preprocessing
# StandardScaler(), KFold()
# selecting model
# Algorithms

if __name__ == '__main__':
    
    #C = 0.001,0.005,0.01,0.05,0.1,0.5,1,2
    C = [1*(10**-5), 1*(10**-4), 1*(10**-3), 0.005,0.001, 0.01,0.05,0.5,
         1,2, 10, 1*(10**2), 1*(10**3), 1*(10**4), 1*(10**5)]
    #factors of ten from 10-7 to 103
    gamma = np.arange(10**(-7),10**3,10)

    """
    The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the train set.
    """
    knn_params = {'classifier': [KNeighborsClassifier()],
                  'classifier__n_neighbors': np.random.randint(0,150,26),
                  'classifier__weights': ['uniform', 'distance'],
                  # eu, eu gain ratio
                  'classifier__matrix': ['euclidean', 'seuclidean','manhattan'],
                  'classifier__algorithm': ['brute']}
                  #'classifier__leaf_size_int': np.random.randint(0, 50, 10)}
    svm_params = {'classifier': [SVC()],
                  'classifier__C': C,
                  'classifier__gamma': gamma,
                  'classifier__kernel': ['rbf', 'linear'],
                  'classifier__strategy': ['mean', 'median']}
    dtr_params = {'classifier': [DecisionTreeClassifier()],
                  'classifier__min_samples_split': range(2, 403, 10),
                  'classifier__criterion': ['gini', 'entropy'],
                  'classifier__max_depth': [2, 4, 6, 8, 10, 12,'auto','sqrt','log2'],
                  'classifier__strategy': ['mean', 'median']}

        # 'most_frequent' or 'constant'
# 2. encode and standarzation
    categorical_transformer = Pipeline(
        steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
               ('encode', OneHotEncoder(handle_unknown='ignore'))
               ])
    numeric_transformer = Pipeline(
        steps=[('miss', SimpleImputer()),  ('scaler', StandardScaler())
               ])
# combine
    preprocessor = ColumnTransformer(transformers=[('categoricals', categorical_transformer, 
                                                    selector(dtype_include="object")),
                                                   ('numericals', numeric_transformer, selector(
                                                       dtype_exclude="object"))
         ])
    prep = [
        {'preprocessing__categoricals': [categorical_transformer],
         'preprocessing__categoricals__miss__strategy': ['most_frequent']},
        {'preprocessing__numericals': [numeric_transformer],
         'preprocessing__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
    ]
    ACC= make_scorer(accuracy_score)
    FSC = make_scorer(f1_score)
    LFT =  make_scorer(recall_score)
    ROC =make_scorer(roc_curve)
    APR = make_scorer(average_precision_score)
    BEP = make_scorer(precision_recall_fscore_support)
    RMS = make_scorer(mean_squared_error)
    MXE = make_scorer(log_loss)
    scorings = [ACC, FSC, LFT, ROC, APR, BEP, RMS, MXE]
    
    ad = pd.read_csv('notebooks/cleaning_data/adult.csv')
    y = ad.target
    X = ad.drop(columns=['target'])
    # make_score_sf(scorings_, df, classifier_, prep_, preprocessor_, train_params)
    pipe_score(scorings,ad,'knn',prep, preprocessor, knn_params)


