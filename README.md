## AUTO GRIDSEARCH-CV!

Still in processing... 

default settings:
- scorings: 
- supervised mechine learning models:
    * preprosessors:categorical/ordinal + numerical variables
    * classifier:
        - SVM
        - KNN
        - K - DTREE
    * Pipeline() :
        pipeline = Pipeline([
            ('preprocessing', preprocessors),
            ('classifier', classifier)])
- useble dataset:
    avl_info={"avl": random_avl()}
    adult_info = {'adult': random_adult()}
    nsr_info = {'nursery': random_nsr()}
    *data_info =  {'dataset_name': [train_set, test_set]}*
- inputs format:
    data_sets = [avl_info, adult_info, nsr_info]

    
        
- paths:
    * models path:
        - all hyperparameters combinations:'results/models/all_models/'
        - best estimator : 'results/models/best_models/'
    * data/training score path: 'results/train/record.json'

HOW TO USE 

from fancy__ import hyper_choice
clf = hyper_choice()
clf.train([Train_sets])
train_

**Limitations NOW :**
1. only suport 3 classifiers
2. not all classifier-score combos (TODO: some will return "ValueErros: not support multiclasses")
3. inputed datasets should in 
    *datasets = [random_adult(n=5000), random_nsr(n=5000)]*
    n is your training size
    TODO: *random_sample(train_size)* still needs to work on to be used for all datasets.

TODO: expectations: 

from fancy__ import hyper_choice
hyper_clf = hyper_choice()
hyper_clf.fit([YOUR_DATASETS])
// automatically train- test -validation your datasets

model  = hyper_clf._best_estimator




