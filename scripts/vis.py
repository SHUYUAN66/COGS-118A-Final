
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
df = pd.DataFrame(random_nsr()[0]['target']).reset_index()

sns.set(
    style="white",
    palette="muted",
    color_codes=True
)
sns.pairplot(df , hue='target', palette='Dark2')
