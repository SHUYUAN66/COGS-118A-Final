import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.manifold import Isomap
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler,  OneHotEncoder, OrdinalEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0, '/scripts')
try:
    from scripts.random_t import *
except Exception:
    from random_t import *
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
C = [ 0.01, 0.5,1, 2 ]
#factors of ten from 10-7 to 103
gamma = [1**(-2), 1, 0, 0.01]

"""
    The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the train set.
    """
knn_params = {'classifier': [KNeighborsClassifier()],
              'classifier__n_neighbors': np.random.randint(1, 50, 10),
              'classifier__weights': ['uniform', 'distance'],
              'classifier__algorithm': ['brute']}
#'classifier__leaf_size_int': np.random.randint(0, 50, 10)}
svm_params = {'classifier': [SVC()],
              'classifier__decision_function_shape': ['ovr'],
              'classifier__C': C,
              'classifier__gamma': gamma,
              'classifier__kernel': ['rbf', 'poly'],
              }
dtr_params = {'classifier': [DecisionTreeClassifier()],
              'classifier__min_samples_split':  [4,7,9],
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_depth': [ 4, 6, 8, 10]
              }

# 'most_frequent' or 'constant'
# 2. encode and standarzation
categorical_transformer = Pipeline(
    steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
           ('encode', OneHotEncoder(handle_unknown='ignore'))
           ])
ordinal_transformer = Pipeline(
    steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
           ('encode', OrdinalEncoder()),
           ])

label_transformer = Pipeline(
    steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
           ('encode', LabelEncoder()),
           ])
numeric_transformer = Pipeline(
    steps=[('miss', SimpleImputer()),  ('scaler', StandardScaler())
           ])
label_transformer = Pipeline(
    steps=[  ('encode', LabelBinarizer())
           ])
# combine
preprocessor = ColumnTransformer(transformers=[('categoricals', categorical_transformer,
                                                selector(dtype_exclude=["int",'float'])),
                                               ('numericals', numeric_transformer, selector(
                                                   dtype_include=["int",'float']))
                                               ])
preprocessor_ordinal = ColumnTransformer(transformers=[('categoricals', ordinal_transformer,
                                                        selector(dtype_exclude=["int", 'float'])),
                                               ('numericals', numeric_transformer, selector(
                                                   dtype_include=["int", 'float']))
                                               ])
preprocessor_label = ColumnTransformer(transformers=[('categoricals', label_transformer
                                                        )
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
prep_label = [
    
]
#multiclass_roc_auc = functools.partial(roc_auc_score, average=np.average)


def multiclass_roc_auc_score(y_test, y_pred, average='micro'):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

ROC = make_scorer(multiclass_roc_auc_score)
ACC = make_scorer(accuracy_score)
PRC = make_scorer(precision_score, average='macro', zero_division=1)
PRC_2 = make_scorer(precision_score, average='micro', zero_division=1)
FSC = make_scorer(f1_score, average='macro' , zero_division=1)
FSC_2 = make_scorer(f1_score, average='micro', zero_division =1)
LFT = make_scorer(recall_score, average='macro', zero_division=1)
LFT_2 = make_scorer(recall_score, average='micro', zero_division =1)

APR = make_scorer(average_precision_score,average='micro',zero_division=1)
APR_2 = make_scorer(average_precision_score, average='macro', zero_division=1)
RMS = make_scorer(mean_squared_error)
MXE = make_scorer(log_loss)

scores_info = {"ROC": ROC,
               'APR': APR,
               'RMS': RMS,
               'MXE': MXE,
               'ACC': ACC,
               'PRC': PRC,
               'FSC': FSC_2,
               'LFT': LFT_2,
               }
# TODO
more_scores = {
                'MEAN':'bb',
              'OPT-SEL':'aa'}

knn_info={'knn':[KNeighborsClassifier(),knn_params]}
svm_info = {'svm': [SVC(), svm_params]}
dtree_info={'dtree':[DecisionTreeClassifier(), dtr_params]}
# Dataset information
# eg: {'avl': [avl_train, avl_test]}
avl_info={"avl": random_avl()}
adult_info = {'adult': random_adult()}
nsr_info = {'nursery': random_nsr()}

data_sets = [avl_info, adult_info, nsr_info]
classifiers = [knn_info, svm_info, dtree_info]

save_trails(prep, preprocessor, svm_info, scores_info, adult_info, path = ['debugging/model_all','debugging/best'])
