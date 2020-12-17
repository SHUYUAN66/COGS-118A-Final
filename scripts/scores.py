import sys
sys.path.insert(0, '/scripts')
from pipelines import *
from sklearn.model_selection import train_test_split, GridSearchCV
from pprintpp import pprint
from sklearn.model_selection import train_test_split, GridSearchCV

# delete later 

# selecting model
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

# accuracy
#Columns = {'accuracy': ACC, 'f1': FSC, 'recall':LFT, MCC, ROC, APR, BEP, RMS, MXE}
def make_score_sf(scorings_,df, classifier_, prep_, preprocessor_, train_params):
    each_score ={}
    X = df.drop(columns=['target'])
    y = df.target
    pipeline, params = make_pipeline(
        classifier_, prep_, preprocessor_,train_params)
    for score in scorings_:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = GridSearchCV(pipeline, params, scoring=score, cv=5,
                           n_jobs=-1, refit=callable, return_train_score=True)
        clf.fit(X_train, y_train)
        print("**pipeline**:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(params)
        print("Best parameters set found on development set:")
        print(" ", clf.best_params_)
        print("Best estimatpr found:")
        print(" ", clf.best_estimator_)
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        each_score[score]=score(y_true,y_pred) 
    return each_score



