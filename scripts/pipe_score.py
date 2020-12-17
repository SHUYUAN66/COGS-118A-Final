from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from pprintpp import pprint
from sklearn.model_selection import KFold, train_test_split, GridSearchCV


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
def pipe_score(scorings_, df, classifier_, prep_, preprocessor_, train_params):
    params = 0
    if classifier_ == 'knn':
        classifier_ = KNeighborsClassifier()
        print('matched')
        prep_.append(train_params)
        
    elif classifier_ == 'svm':
        classifier_ = SVC()
        prep_.append(train_params)
        
    elif classifier_ == 'dtree':
        classifier_ = DecisionTreeClassifier()
        prep_.append(train_params)
        
    params = prep_
    
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', classifier_)])
    # next
    each_score = {}
    X = df.drop(columns=['target'])
    y = df.target
    for score in scorings_:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        print(pipeline)
        clf = GridSearchCV(pipeline, params, scoring=score, cv=3,
                           n_jobs=-1, refit=callable, return_train_score=True)
        clf.fit(X_train, y_train)
        print("**pipeline**:", [name for name, _ in pipeline.steps])
        print()
        print("parameters:")
        print(params)
        print()
        print("Best parameters set found on development set:")
        print(" ", clf.best_params_)
        print()
        print("Best estimator found:")
        print(" ", clf.best_estimator_)
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        each_score[score] = recall_score(y_true, y_pred)
    return each_score
