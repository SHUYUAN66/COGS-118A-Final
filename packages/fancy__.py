
from packages.check_status import check_directory
from sklearn.preprocessing import LabelBinarizer
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os
import json
__doc__

'''Traininging......... '''
# input your own pipelines!
class hyper_choice:

    def __init__(self, pre_params_, preprocessor_,alfs,srcs ,n):
        self._pre_params_ = pre_params_
        self._preprocessor_ = preprocessor_
        self._alfs = alfs
        self._scrs = srcs
        self._train_size = n # will work later
        self._models_path = ['results/models/all_models/','results/models/best_models/']
        self._data_path = "results/train/record.json"
        self._trailnum = 0

    
    # specific alg-scr-data combo 
    # sve_trails(classifier, dataset)
    def _save_trails(self, alg, data):
        path = self._models_path
        alg_name = list(alg)[0]
        data_name = list(data)[0]
        print(data_name)
        clsf = alg[alg_name][0]
        param = alg[alg_name][1]
        params = self._pre_params_
        params.append(param)
        dataset = data[data_name][0]
        pipeline = Pipeline([
            ('preprocessing', self._preprocessor_),
            ('classifier', clsf)])
        X = dataset.drop(columns=['target']) 
        y = dataset.target
        # except adult-ROC combo TODO: fix this
        # use if
        if len(y.unique()) > 2:
            #run when multi-classes
            lb = LabelBinarizer().fit(y)
            y = lb.transform(y)
   
        record_scores={}
        scr = self._scrs
        for i in range(len(list(self._scrs))):  
            score_name = list(scr)[i]
            score = scr[score_name]
            print("Evaluation score is : ", score_name.upper()) 
            record_scores[score_name] ={}
            score_details = {}
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2)
            clf = GridSearchCV(pipeline, params, scoring=score,
                           cv=5, n_jobs=-1, return_train_score=True, verbose=True)
            clf.fit(X_train, y_train)
            save_models = os.path.join(path[0],alg_name, data_name, score_name)
            check_directory([save_models])
            joblib.dump(clf, os.path.join(
                save_models, 'all_models.pkl'))
            print('All models scored in ' +score_name+ ' saved in ', save_models)
            results = clf.cv_results_
            mean_train_grade = clf.cv_results_['mean_test_score']
            best_score = clf.best_score_
            best_params = clf.best_params_
            best_model = clf.best_estimator_
            score_details['mean_train_score'] = mean_train_grade
            score_details['best_score'] = best_score
            score_details['best_params'] = best_params
            score_details['results'] = results
            save_best_model = os.path.join(
                path[1], alg_name, data_name, score_name)
            check_directory([save_best_model])
            score_details['best_estimator'] = best_model
            joblib.dump(best_model, os.path.join(
                save_best_model, 'best_model.pkl'))
            print('Best estimator scored in ' +score_name+ ' saved in ', save_best_model)
            record_scores.update(score_details)

        return record_scores
        
    # loop though
    
    def train(self, datasets):
        recordings = {}
        #model_path = self._models_path
        for i in self._alfs:
            print('START ', list(i)[0].upper(), 'Classification')
            for j in datasets:
                # sve_trails(classifier, dataset)
                self._save_trails(i, j)
                    # (self, alg, scr, data,path)
                    #self._pre_params_, self._preprocessor_, i, self._scrs, j, self._models_path)
            print('FINISH ', i)
            print('')
        json = json.dumps(recordings)
        check_directory([self._data_path])
        f = open(self._data_path, "w")
        f.write(json)
        f.close()
        print('FINISHED ALLLLLLLLLL')
        return


    def test(alfs, datasets,scrs):
    # get algorithm:
        return
    

    

    






