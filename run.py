from packages.fancy__ import hyper_choice
from scripts.random_t import *
from scripts.main import *

if __name__ == '__main__':
    clf = hyper_choice(prep, preprocessor, classifiers,scores_info, 5000)
    clf.train(data_sets)
