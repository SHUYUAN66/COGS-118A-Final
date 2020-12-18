import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        if not CHECK_FOLDER:
            os.makedirs(i)     
        else:
            pass
def got_pipeline(alg, pre_params_, preprocessor_):
    alg_name = list(alg)[0]
    clsf = alg[alg_name][0]
    param = alg[alg_name][1]
    params = pre_params_.append(param)
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', clsf)])
    return pipeline, params

def save_trails(pipeline_, params_, alg, scr, data, path=['all_models', 'best_models']):
    """
    alg: Algorithm, a dict(), each one is a name of that function {'knn':[KNN(),knn_params]}
    scr: scorings, a dict(), each one is a function such as 'acc':ACC 
    data: a dict() of dataset {'dataname':[trainset, testset]}
    path = ['all_models','best_models'] , to decide first chart or second chart.
    """
    # parameters
    alg_name = list(alg)[0]
    data_name = list(data)[0]
    
    dataset = data[data_name][0]
    X = dataset.drop(columns=['target'])
    y = dataset.target
    record={}
    for i in range(len(list(scr))):  
        score_name = list(scr)[i]
        score = scr[score_name]

        trail_name = alg_name+data_name+score_name
        record[trail_name] =[]
        details = {}
        # 4000 : 1000
        X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.2)
        clf = GridSearchCV(pipeline_, params_, scoring=score, cv=5, n_jobs=-1, return_train_score=True)
        clf.fit(X_train, y_train)
        # For each trialwe use 4000 cases to train thedi erent models,1000 casesto calibrate the models and select the best parameters,and then report performance on the large final test set.
        save_models = os.path.join(path[0], alg_name, data_name, score_name)
        check_directory(save_models)
        joblib.dump(clf, os.path.join(
            save_models, 'all_models.pkl'))
        mean_train_grade = clf.cv_results_['mean_test_score']
        best_score = clf.best_score_
        best_params = clf.best_params_
        best_model = clf.best_estimator_
        details['mean_train_score'] = mean_train_grade
        details['best_score'] = best_score
        details['best_params'] = best_params
        save_best_model = os.path.join(
            path[1], alg_name, data_name, score_name)
        check_directory(save_best_model)
        details['best_estimator'] = best_model
        joblib.dump(best_model, os.path.join(
            save_best_model, 'best_model.pkl'))
        record[trail_name].append(details)
    return 

'''
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
'''
