
import sys
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, '/scripts')
from save import *
# Basic
from sklearn.metrics import  make_scorer, f1_score, accuracy_score, mean_squared_error, average_precision_score, roc_auc_score, log_loss, recall_score,precision_score

"""
* The DataFrame: 
    group1: ACC, FSC, LFT
    group2: ROC, APR, BEP
    group3: RMS, MXE
    group4: OPT-SELS

refer: https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn

* Within the Matric: 
    - The algorithm with the best performance on each metric is ***boldfaced***. Using t-test p= 0.05, others are *'ed in three trails if they still have good performance.
    
"""
   #C = 0.001,0.005,0.01,0.05,0.1,0.5,1,2
C = [1*(10**-5), 1*(10**-4), 1*(10**-3), 0.005, 0.001, 0.01, 0.05, 0.5,
     1, 2, 10, 1*(10**2), 1*(10**3), 1*(10**4), 1*(10**5)]
#factors of ten from 10-7 to 103
gamma = np.arange(10**(-7), 10**3, 10)

"""
    The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the train set.
    """
knn_params = {'classifier': [KNeighborsClassifier()],
              'classifier__n_neighbors': np.random.randint(1,150, 26),
              'classifier__weights': ['uniform', 'distance'],
              # eu, eu gain ratio
              #'classifier__matrix': ['euclidean', 'seuclidean','manhattan'],
              'classifier__algorithm': ['brute']}
#'classifier__leaf_size_int': np.random.randint(0, 50, 10)}
svm_params = {'classifier': [SVC()],
              'classifier__C': C,
              'classifier__gamma': gamma,
              'classifier__kernel': ['rbf', 'linear'],
              'classifier__strategy': ['mean', 'median']}
dtr_params = {'classifier': [DecisionTreeClassifier()],
              'classifier__min_samples_split':  np.random.randint(1,150, 5),
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_depth': [2, 4, 6, 8, 10, 12, 'auto', 'sqrt', 'log2'],
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
ACC = make_scorer(accuracy_score)
PRC = make_scorer(precision_score)
FSC = make_scorer(f1_score)
LFT = make_scorer(recall_score)
ROC = make_scorer(roc_auc_score)
APR = make_scorer(average_precision_score)
RMS = make_scorer(mean_squared_error)
MXE = make_scorer(log_loss)

scorings = [ACC, PRC, FSC, LFT, ROC, APR, RMS, MXE]
name = ['ACC', 'PRC','FSC', 'LFT', 'ROC', 'APR', 'RMS', 'MXE']
ad_train = pd.read_csv('data/train/adult.csv')
nsr_train = pd.read_csv('data/train/nsr.csv')
avl_train = pd.read_csv('data/train/avl.csv')
ad_test = pd.read_csv('data/test/adult.csv')
nsr_test = pd.read_csv('data/test/nsr.csv')
avl_test = pd.read_csv('data/test/avl.csv')

scores_info = {'ACC':ACC,
               'PRC':PRC,
               'FSC':FSC,
               'LFT':LFT,
               'ROC':ROC,
               'APR':APR,
               'RMS':RMS,
               'MXE':MXE}
# TODO!
more_scores = {'MEAN':'bb',
              'OPT-SEL':'aa'}

knn_info={'knn':[KNeighborsClassifier(),knn_params]}
svm_info={'svm':[SVC(),svm_params]}
dtree_info={'dtree':[DecisionTreeClassifier(),dtr_params]}
avl_info={"avl":[avl_train,avl_test]}
adult_info ={'adult':[ad_train, ad_test]}
nsr_info ={'nursery':[nsr_train, nsr_test]}

pipeline, spacing = got_pipeline(knn_info, prep, preprocessor)

save_trails(pipeline, spacing, knn_info, scores_info, nsr_info)

