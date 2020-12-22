# TODO: establish chart for score- classifier comparisons
import os
import numpy as np
import os.path
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error, average_precision_score, roc_auc_score, log_loss, recall_score, precision_score
import joblib
from sklearn.model_selection import train_test_split
record = {}
# measure model performance on the test set (all the data in thedataset other than the 5000 random samples
# for each dataset, generate all scores of "score"
# need to record as test_scores = {adult:{svc:[], knn:[],dtree:[]}, nursery: {svc:[], knn:[],dtree:[]}, power: {svc:[], knn:[],dtree:[]}
scores_test_alg = {}
# mean​ training​ set performance for the optimal hyperparameters
# a good hyper, then use these params to train and got scores
# this would be only using hyperparams model:
# record as model:{'adult:{best = pipeline,test = [], train = []}, nursery:{test = [], train = []}...}
scores_train_data = {}
# a discussion of thedifference between each algorithms’ training and test set performance
# continue on usinf the same good params, see the test scores
scores_test_data = {}
ad_train = pd.read_csv('data/train/adult.csv')
nsr_train = pd.read_csv('data/train/nsr.csv')
avl_train = pd.read_csv('data/train/avl.csv')


"""
for every algorithms (knn/svm/dtree)
there are there three datasets and many scores evaluating them.
find all the test score of such dataset predicted from algorithm based on scorring,
mean[ad-svm-acc,nsr_svm_acc, avl_svm_acc] (1,1) 
mean[ad-knn-acc,nsr_svm_acc,avl_nsr_acc] (2,1)
mean[ad-dtree-acc,nsr_dtree_acc,avl_dtree_acc](2,1)
"""

ACC = make_scorer(accuracy_score)
PRC = make_scorer(precision_score)
FSC = make_scorer(f1_score)
LFT = make_scorer(recall_score)
ROC = make_scorer(roc_auc_score)
APR = make_scorer(average_precision_score)
RMS = make_scorer(mean_squared_error)
MXE = make_scorer(log_loss)
# score 
scorings = [ACC, PRC, FSC, LFT, ROC, APR, RMS, MXE]
name = ['ACC', 'FSC', 'LFT', 'ROC', 'APR', 'BEP', 'RMS', 'MXE']
# score name score.get_key() 
score_dict={}
for i in range(len(name)):
    score_dict[name[i]] = scorings[i]
# algorithm


def select_trails(alg, scr, data, path=['all_models', 'best_models']):
    """
    alg: Algorithm, a list of names}
    scr: scorings, a dict(), each one is a function such as 'acc':ACC 
    data: a dict() of dataset 
    path = ['all_models','best_models'] , to decide first chart or second chart.
    """
    data_name = list(data)[0]
    dataset = data[data_name][0]
    X_test = dataset.drop(columns=['target'])
    y_test = dataset.target
    
    
    

    

    
    


