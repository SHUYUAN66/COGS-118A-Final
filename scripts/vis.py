
import pandas as pd
import numpy as np
sys.path.insert(0, '/scripts')
from random_t import *
from save import *
# Dataset information 
# eg: {'avl': [avl_train, avl_test]}
avl_info = {"avl": random_avl()}
adult_info = {'adult': random_adult()}
nsr_info = {'nursery': random_nsr()}
knn_info = {'knn': [KNeighborsClassifier(), knn_params]}
svm_info = {'svm': [SVC(), svm_params]}
dtree_info = {'dtree': [DecisionTreeClassifier(), dtr_params]}

# based on different hyperparams | different train set (change n)
# only choose the best 

def compare_test(alg_info, data_info):


