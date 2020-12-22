from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler  
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

def pre_encoder(x):
    str_encode = Pipeline(
        steps=[('miss', SimpleImputer(strategy='constant', fill_value='missing')),
               ('encode', x)
               ])
    num_encode = Pipeline(
        steps=[('miss', SimpleImputer()),  ('scaler', StandardScaler())
               ])

    pre_encode = ColumnTransformer(transformers=[('categoricals', str_encode,
                                                  selector(dtype_exclude=['float'])),
                                                 ('numericals', num_encode, selector(
                                                     dtype_include=['float']))
                                                 ])
    parameters = [
        {'pre_encode__categoricals': [str_encode],
         'pre_encode__categoricals__miss__strategy': ['most_frequent']},
        {'pre_encode__numericals': [num_encode],
         'pre_encode__numericals__miss__strategy': ['most_frequent'],
         'pre_encode__numericals__miss__strategy': ['mean', 'median', 'most_frequent']}
    ]
    # return preprocessor and their parameters
    return [parameters, pre_encode]


def multiclass_roc_auc_score(y_test, y_pred, average='micro'):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
