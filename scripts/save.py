import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import label_binarize

def check_directory(lst):
    for i in lst:
        CHECK_FOLDER = os.path.isdir(i)
        if not CHECK_FOLDER:
            os.makedirs(i)     
        else:
            return
def save_trails(pre_params_, preprocessor_, alg, scr, data, path=['all_models/', 'best_models/']):
    """
    alg: Algorithm, a dict(), each one is a name of that function {'knn':[KNN(),knn_params]}
    scr: scorings, a dict(), each one is a function such as 'acc':ACC 
    data: a dict() of dataset {'dataname':[trainset, testset]}
    path = ['all_models','best_models'] , to decide first chart or second chart.
    """
    
    # parameters
    params = pre_params_
    alg_name = list(alg)[0]
    data_name = list(data)[0]
    print(data_name)
    clsf = alg[alg_name][0]
    param = alg[alg_name][1]
    params.append(param)
    dataset = data[data_name][0]
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', clsf)])
    #  ROC AUC requires the predicted class probabilities (yhat_probs)
    X = dataset.drop(columns=['target']) 
    y = dataset.target
    y = label_binarize()(y)
    record={}
    for i in range(len(list(scr))):  
        score_name = list(scr)[i]
        score = scr[score_name]
        print(score_name)
        trail_name = alg_name+data_name+score_name
        record[trail_name] =[]
        details = {}
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.5,random_state=13)
        print(y_train.unique())
        print(y_val.unique())
        clf = GridSearchCV(pipeline, params, scoring=score,
                           cv=5, n_jobs=-1, return_train_score=True, verbose=True)
        # For each trialwe use 4000 cases to train thedi erent models,1000 casesto calibrate the models and select the best parameters,and then report performance on the large final test set.
        clf.fit(X_train, y_train)
        save_models = os.path.join(path[0],alg_name, data_name, score_name)
        check_directory([save_models])
        joblib.dump(clf, os.path.join(
            save_models, 'all_models.pkl'))
        results = clf.cv_results_
        mean_train_grade = clf.cv_results_['mean_test_score']
        print('almost there...!')
        best_score = clf.best_score_
        best_params = clf.best_params_
        best_model = clf.best_estimator_
        details['mean_train_score'] = mean_train_grade
        details['best_score'] = best_score
        details['best_params'] = best_params
        save_best_model = os.path.join(
            path[1], alg_name, data_name, score_name)
        check_directory([save_best_model])
        details['best_estimator'] = best_model
        joblib.dump(best_model, os.path.join(
            save_best_model, 'best_model.pkl'))
        record[trail_name].append(details)
    
    return record

