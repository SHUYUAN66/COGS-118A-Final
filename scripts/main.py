

import functools
import sys
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import LabelBinarizer, StandardScaler,  OneHotEncoder, OrdinalEncoder
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
C = [0.005, 0.001, 0.01, 0.05, 0.5,1, 2 ]
#factors of ten from 10-7 to 103
gamma = [10**(-5), 10, 1, 0, 0.01]

"""
    The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the train set.
    """
knn_params = {'classifier': [KNeighborsClassifier()],
              'classifier__n_neighbors': np.random.randint(1, 50, 10),
              'classifier__weights': ['uniform', 'distance'],
              'classifier__algorithm': ['brute']}
#'classifier__leaf_size_int': np.random.randint(0, 50, 10)}
svm_params = {'classifier': [SVC()],
              'classifier__C': C,
              'classifier__gamma': gamma,
              'classifier__kernel': ['rbf', 'poly'],
              }
dtr_params = {'classifier': [DecisionTreeClassifier()],
              'classifier__min_samples_split':  [4,7,9],
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_depth': [ 4, 6, 8,  10]
              }

# 'most_frequent' or 'constant'
# 2. encode and standarzation
categorical_transformer = Pipeline(
    steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
           ('encode', OneHotEncoder(handle_unknown='ignore'))
           ])
ordinal_transformer = Pipeline(
    steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
           ('encode', OrdinalEncoder())
           ])
numeric_transformer = Pipeline(
    steps=[('miss', SimpleImputer()),  ('scaler', StandardScaler())
           ])
# combine
preprocessor = ColumnTransformer(transformers=[('categoricals', categorical_transformer,
                                                selector(dtype_include=["object","category"])),
                                               ('numericals', numeric_transformer, selector(
                                                   dtype_include=["int",'float']))
                                               ])
preprocessor_ordinal = ColumnTransformer(transformers=[('categoricals', ordinal_transformer,
                                                        selector(dtype_exclude=["int"])),
                                               ('numericals', numeric_transformer, selector(
                                                   dtype_include=["int", 'float']))
                                               ])
prep = [
    {'preprocessing__categoricals': [categorical_transformer],
     'preprocessing__categoricals__miss__strategy': ['most_frequent']},
    {'preprocessing__numericals': [numeric_transformer],
     'preprocessing__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
]
prep_ord = [
    {'preprocessing__categoricals': [ordinal_transformer],
     'preprocessing__categoricals__miss__strategy': ['most_frequent']},
    {'preprocessing__numericals': [numeric_transformer],
     'preprocessing__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
]
prep_cat = [
    {'preprocessing__categoricals': [categorical_transformer],
     'preprocessing__categoricals__miss__strategy': ['most_frequent']},
    {'preprocessing__numericals': [numeric_transformer],
     'preprocessing__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
]
#multiclass_roc_auc = functools.partial(roc_auc_score, average=np.average)
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

ROC = make_scorer(multiclass_roc_auc_score)
ACC = make_scorer(accuracy_score)
PRC = make_scorer(precision_score, average ='micro' )
PRC_2 = make_scorer(precision_score, average='macro')
FSC = make_scorer(f1_score, average='macro')
LFT = make_scorer(recall_score, average='macro')
LFT_2 = make_scorer(recall_score, average='micro')
APR = make_scorer(average_precision_score,average='micro')
APR_2 = make_scorer(average_precision_score, average='macro')
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

scores_info = {"ROC": ROC,
               'APR': APR,
               'RMS': RMS,
               'MXE': MXE,
               'ACC': ACC,
               'PRC': PRC,
               'FSC': FSC,
               'LFT': LFT,
               }
# TODO!
more_scores = {
                'MEAN':'bb',
              'OPT-SEL':'aa'}

knn_info={'knn':[KNeighborsClassifier(),knn_params]}
svm_info = {'svm': [SVC(), svm_params]}
dtree_info={'dtree':[DecisionTreeClassifier(), dtr_params]}
avl_info={"avl":[avl_train,avl_test]}
adult_info ={'adult':[ad_train, ad_test]}
nsr_info ={'nursery':[nsr_train, nsr_test]}

# save_trails(pre_params_, preprocessor_, alg, scr, data)
#nsr_knn=save_trails(prep, preprocessor, knn_info, scores_info, nsr_info)
#adult_knn = save_trails(prep, preprocessor, knn_info, scores_info, adult_info)
print('finish knn')
print('start_svm')
#nsr_svm=save_trails(prep, preprocessor, svm_info, scores_info, nsr_info)
#save_trails(prep, preprocessor, svm_info, scores_info, adult_info)
print('finish svm')
print('start dtree')
save_trails(prep, preprocessor, dtree_info, scores_info, adult_info)
#nsr_dtree=save_trails(prep, preprocessor, dtree_info, scores_info, nsr_info)
print('finished dtree' )
