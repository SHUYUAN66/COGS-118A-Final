import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import LabelBinarizer
from preprocessing import *

def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        if not CHECK_FOLDER:
            os.makedirs(i)     
        else:
            return


def save_trails(encoder, alg, scr, data, path):
    """
    pre_parameters= pre_encoder(encoder)[0]
    preprocessor =pre_encoder(encoder)[1]
    alg: Algorithm, a dict(), each one is a name of that function {'knn':[KNN(),knn_params]}
    scr: scorings, a dict(), each one is a function such as 'acc':ACC 
    data: a dict() of dataset {'dataname':[trainset, testset]}
    path = ['~/all_models','~/best_models'] , to decide first chart or second chart.
    """
    # parameters
    params = pre_encoder(encoder)[0]
    preprocessor_ = pre_encoder(encoder)[1]
    alg_name = list(alg)[0]
    data_name = list(data)[0]
    clsf = alg[alg_name][0]
    param = alg[alg_name][1]
    params.append(param)
    dataset = data[data_name][0]
    test_set = data[data_name][1]
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', clsf)])
    X = dataset.drop(columns=['target']) 
    y = dataset.target
    print(y.unique())
    if len(y.unique()) > 2:
        # this will transform y into 1,0 ndarray
        lb = LabelBinarizer().fit(y)
        y = lb.transform(y)
       
    record_scores={}
    for i in range(len(list(scr))): 
        score_name = list(scr)[i]
        score = scr[score_name]
        print("Dataset is : ", data_name.upper()) 
        record_scores[score_name] ={}
        score_details = {}
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2)
        clf = GridSearchCV(pipeline, params, scoring=score,
                           cv=5, n_jobs=-1,refit=True, return_train_score=True, verbose=True)
        clf.fit(X_train, y_train)
        # test dataset
        best_model = clf.best_estimator_
        X_test = test_set.drop(columns = ['target'])
        y_test = test_set.target
        y_pred = best_model.predict(X_test)
        testset_score = (y_test,y_pred)
        score_details['testset_score'] = testset_score
        score_details['best_score'] = clf.best_score_
        score_details['results'] = clf.cv_results_


        # save models to path
        save_models = os.path.join(path[0], alg_name, data_name, score_name)
        save_best_model = os.path.join(path[1], alg_name, data_name, score_name)
        check_directory([save_best_model, save_models])
        joblib.dump(clf, os.path.join(
            save_models, 'all_models.pkl'))
        joblib.dump(best_model, os.path.join(
            save_best_model, 'best_model.pkl'))
        print('All models scored in ' + score_name + ' saved in ', save_models)
        print('Best model scored in ' +score_name+ ' saved in ', save_best_model)
        record_scores.update(score_details)
    print(record_scores)
    return record_scores

