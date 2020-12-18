import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

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
def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(i)
            print("created folder : ", i)
        else:
            print(i, "folder already exists.")

def pipe_score(scorings_, df, classifier_, prep_, preprocessor_, train_params, path_):
    way = classifier_
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
    name = ['ACC', 'FSC', 'LFT', 'ROC', 'APR', 'BEP', 'RMS', 'MXE']
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', classifier_)])
    # next
    X = df.drop(columns=['target'])
    y = df.target
    for i in range(len(scorings_)):
        
        score= scorings_[i]
        print("# Tuning hyper-parameters for %s" % score)
        print()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        print(pipeline)
        clf = GridSearchCV(pipeline, params, scoring=score, cv=5,
                           n_jobs=-1, refit=callable, return_train_score=True)
        clf.fit(X_train, y_train)
        # only save best hyperparmas model
        print('processing to save all.....')
        
        save_all_on_alg = os.path.join(path_, name[i])
        save_all_on_data = os.path.join('models/', way)
        save_good_alg =os.path.join('models/best_models', way)
        
        lst_ = [save_all_on_alg, save_all_on_data, save_good_alg]
        check_directory(lst_)
        joblib.dump(clf, os.path.join(save_all_on_alg, way+'_all.pkl'))
        joblib.dump(clf, os.path.join(save_all_on_data, name[i]+'_all.pkl'))
        print('saved all models')
        print('')
        print('search best model...')
        # save all models
        print(clf.return_train_score,";")
        joblib.dump(clf.best_estimator_, os.path.join(
            save_good_alg, name[i]+'best.pkl'))
        print("SAVE! ")
        print()
    return