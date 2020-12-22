from logging import captureWarnings
import warnings

from sklearn.model_selection import GridSearchCV
from packages.fancy__ import hyper_choice
import sys
sys.path.insert(0,'/scripts')
from scripts.random_t import *
from scripts.main import *


if __name__ == '__main__':
    """
    bugs:
    1. [SMV-AVL-LFT]: 
    bugs1 if len>2: ValueError: y should be a 1d array, got an array of shape (3200, 12) instead.warnings.warn("Estimator fit failed. The score on this train-test")

    bugs2 comment if len>2: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.

    solution: use ordinalencoder in preprossor instead. [because of SVM only for 1d?]
    2. [svm/nursery/APR]: solved by OneVsRestClassifier]
    3. [svm/nursery/RMS]

    """
    prep = prep_ord
    classifiers = [svm_info]
    scores_info = {'APR': APR}
    data_sets = [nsr_info]
    
    # pre_params_, preprocessor_,alfs,srcs ,n
    # dict_keys(['add_indicator', 'copy', 'fill_value', 'missing_values', 'strategy', 'verbose']) dict_keys(['categories', 'drop', 'dtype', 'handle_unknown', 'sparse'])
    clf = hyper_choice(prep_ord,preprocessor_ordinal,classifiers, scores_info, 3000)
    # alg_name, data_name, score_name
    clf.train(data_sets)

    
