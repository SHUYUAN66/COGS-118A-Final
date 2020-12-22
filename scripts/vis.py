
import seaborn as sns
import pandas as pd
import sys
sys.path.insert(0, '/scripts')
from random_t import *
from save import *
from main import *
# Dataset information 
# eg: {'avl': [avl_train, avl_test]}

# based on different hyperparams | different train set (change n)
# only choose the best 

from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG
import  jolib
test_this = joblib.load('models/best_models/knn/ault/ACC/best_model.pkl')
train = adult_info()[0]
test = adult_info()[1]
print(test_this)
X_train = train.drop(columns=['target'])
y_train = train.target
#test_this.fit(X_train, y_train)
viz = dtreeviz(test_this,
               X_train,
               y_train,
               target_name='label',  # this name will be displayed at the leaf node
               feature_names=train.feature_names,
               fontname="Arial",
               title_fontsize=16,
               colors = {"title":"purple"}
              )
