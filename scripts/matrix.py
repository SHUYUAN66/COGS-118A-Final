import glob
import os
from os.path import dirname, basename, isfile, join
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error, average_precision_score, roc_auc_score, log_loss, recall_score, precision_score
import joblib
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

ad_test = pd.read_csv('data/test/adult.csv')
nsr_test = pd.read_csv('data/test/nsr.csv')
avl_test = pd.read_csv('data/test/avl.csv')
"""
for every algorithms (knn/svm/dtree)
there are there three datasets and many scores evaluating them.
find all the test score of such dataset predicted from algorithm based on scorring,
mean[ad-svm-acc,nsr_svm_acc, avl_svm_acc] (1,1) 
mean[ad-knn-acc,nsr_svm_acc,avl_nsr_acc] (2,1)
mean[ad-dtree-acc,nsr_dtree_acc,avl_dtree_acc](2,1)
"""
# one test set: 
# knn (ACC, APR, .....)
# - ad
# - nsr
# - avl 
# svm
# ...
ACC = make_scorer(accuracy_score)
PRC = make_scorer(precision_score)
FSC = make_scorer(f1_score)
LFT = make_scorer(recall_score)
ROC = make_scorer(roc_auc_score)
APR = make_scorer(average_precision_score)
RMS = make_scorer(mean_squared_error)
MXE = make_scorer(log_loss)


path = 'models'
def collect(dire,test,path):

    scorings = [ACC, PRC, FSC, LFT, ROC, APR, RMS, MXE]
    name = ['ACC', 'FSC', 'LFT', 'ROC', 'APR', 'BEP', 'RMS', 'MXE']
    y_true = 0
    for i in range(len(name)):
        name = name[i]
        exc = scorings[i]
        model_name = os.path.join(path,name+"_all.pkl")
        model = joblib.load(model_name)
        dire[name] = model
        x_test = test.drop(columns=['target'])  # df.drop(columns=['target'])
        y_true = test.target
        #dire[name+"_score"] = model.scorer_['mean_test_score']
        print('pred', model.predict(x_test))
        y_true = list(y_true)
        pred = model.predict(x_test)
        dire[name+"_test_score"] = exc(y_true,pred )
        
    print(dire)
    return dire


test = pd.read_csv('data/test/adult.csv')
knn={}
collect(knn, test, path='models/knn')


