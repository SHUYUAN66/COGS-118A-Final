# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:32:47 2020

@author: aparn
"""


'''
HELPFUL LIBRARIES
numpy
scikit - this is where most of the things you need to implement is available
       - StandardScaler(), KFold(), SVM() ..... all models
       - auc(), fscore(), confusion_matrix()
matplotlib - plot() etc...
'''

import numpy as np

x = np. random.rand(5000,10)
y = np.random.randint(0,2,(5000,1)) # binary label

print(x.shape,y.shape)

n_folds = 5
n_per_fold = int(len(x)/n_folds)

fold_index = list(np.arange(5))

x_fold = []
y_fold = []

for i in fold_index:
    print(i*n_per_fold,(i*n_per_fold)+n_per_fold)
    x_fold.append(x[i*n_per_fold:(i*n_per_fold)+n_per_fold])
    y_fold.append(y[i*n_per_fold:(i*n_per_fold)+n_per_fold])
    
    
for i in range(n_folds):
    x_test = x_fold[i]
    y_test = y_fold[i]
    
    fold_index = list(np.arange(5))
    fold_index.remove(i)
    
    x_train = np.empty((0,10))
    y_train = np.empty((0,1))
    
    for j in fold_index:
        x_train = np.append(x_train,x_fold[j],0)
        y_train = np.append(y_train,y_fold[j],0)
        
    print('fold',i,'test',x_test.shape,y_test.shape,'train',x_train.shape,y_train.shape)
    
    
    #--------
    #normalize
    #---------
    
    
    #-----------------------------
    #BELOW IS A PSEUDOCODE FOR HOW THE REST OF THE PROJECT WOULD TYPICALLY FOLLOW
    #------------------------------
    #for each trial
    n_trials = 3
    for k in range(n_trials):
        para1 = 1 # !
        para2 = 2 # !
        #grid search for hyperparameters
        for p1 in para1:
            for p2 in para2:
                # !
                # !
                pass
                
    pass             
                #fitted_model = model(x_train,y_train,parameters = (p1,p2,...))
                
                #pkl.dump(fitted model)
                
                #TIP: train and save your model with correct names; so that all you need to do is test later on
                
                
                #testing
                #y_pred = model.predict(x_test)
                
                #compute metrics
                #AUC, F-Score, Confusion #whatever metric you wish to report
                #use sklearn 
                
                
                #TIP: save everything! models in particular & the results so that you can generate results quickly
                
                
