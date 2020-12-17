
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_pipeline(classifier_, prep_, preprocessor_, train_params):
    # missing values
    # parameters:
    model =0
    if classifier_== 'knn':
        classifier_ = KNeighborsClassifier()
        prep_.append(train_params)
    elif classifier_ == 'svm':
        classifier_ = SVC()
        prep_.append(train_params)
    elif classifier_ == 'dtree':
        classifier_ = DecisionTreeClassifier()
        prep_.append(train_params)
    pipeline = Pipeline([
        ('preprocessing', preprocessor_),
        ('classifier', model)
    ])
    return pipeline, prep_


