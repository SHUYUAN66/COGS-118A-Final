
import json
import pandas as pd
import numpy as np
from pandas import CategoricalDtype
def incomeFixer(x):
    if x == "<=50K":
        return 0
    else:
        return 1


def clean_adult(df):
    ad_var = ['age', 'workclass', 'fnlwgt', 'education',
              'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    df.columns = ad_var
    copy = df.copy()
    copy = copy.replace({' ?': np.nan, 0: np.nan, 99999: np.nan})
    copy = copy.replace(" ", "", regex=True)
    copy = copy.replace("-", "", regex=True)
    # f.replace('_', '', regex=True)
    copy["target"] = copy.apply(lambda x: incomeFixer(x['target']), axis=1)
    cp_dp = copy.copy()
    cp_dp = cp_dp.drop(
        columns=['education', 'fnlwgt', 'capital-gain', 'capital-loss'])
    DataFrame = cp_dp.copy()
    categorical_missing = ['workclass', 'occupation', 'native-country']
    for ColName in categorical_missing:
        most_frequent_category = DataFrame[ColName].mode()[0]
    # replace nan values with most occured category
        DataFrame[ColName + "_Imputed"] = DataFrame[ColName]
        DataFrame[ColName +
                  "_Imputed"].fillna(most_frequent_category, inplace=True)
        DataFrame[ColName] = DataFrame[ColName + "_Imputed"]
        DataFrame = DataFrame.drop([ColName + "_Imputed"], axis=1)
    return DataFrame


def random_adult(n=5000):
    adult_raw = pd.read_csv('./data/raw/adult/adult.data', header=None)
    cleaning_adult = clean_adult(adult_raw)
    adult_train = cleaning_adult.sample(n=n, replace=False)
    adult_test = cleaning_adult.drop(adult_train.index)
    adult_train.to_csv('./data/train/adult.csv', index=False)
    adult_test .to_csv('./data/test/adult.csv', index=False)
    adult_train = pd.read_csv('./data/train/adult.csv')
    adult_test = pd.read_csv('./data/train/adult.csv')
    return [adult_train, adult_test]


with open("./scripts/nsrorder.json", "r") as read_file:
    print("Converting JSON encoded data into Python dictionary")
    developer = json.load(read_file)
    print("Decoded JSON Data From File")
    for key, value in developer.items():
        print(key, ":", value)
    print("Done reading json file")


def check_item(raw, in_r):
    for i in in_r:
        if i in raw:
            pass
        else:
            print('wrong item is : ', i)
    #return'No Error Found in column '


def clean_nsr(df):
    od = developer
    nsr_var = ['parents', 'has_nurs', 'form', 'children',
               'housing', 'finance', 'social', 'health', 'target']
    df.columns = nsr_var
    raw = df.copy()
    raw = raw.replace({'inconv': 0, 'convenient': 1})
    df = df.replace('_', '', regex=True)
    df = df.drop(columns=['finance'])
    for i in df.columns:
        df[i] = df[i].astype('category')
        r = od[i]
        cat_r = CategoricalDtype(categories=r, ordered=True)
        # give the order
        df[i] = df[i].cat.reorder_categories(r, ordered=True)

    df['finance'] = raw['finance']
    return df

def random_nsr(n=5000):   
    nsr_raw = pd.read_csv('./data/raw/nursey/nursery.data', header=None)
    cleaning_nsr = clean_nsr(nsr_raw)
    cleaning_nsr['target'] = cleaning_nsr['target'].replace(
        ['notrecom', 'recommend', 'veryrecom', 'priority', 'specprior'], [0, 0, 2, 3, 4])
    nsr_train = cleaning_nsr.sample(n=n, replace=False)
    nsr_test = cleaning_nsr.drop(nsr_train.index)
    nsr_train.to_csv('./data/train/nsr.csv', index=False)
    nsr_test .to_csv('./data/test/nsr.csv', index=False)
    nsr_train = pd.read_csv('./data/train/nsr.csv')
    nsr_resr = pd.read_csv('./data/train/nsr.csv')
    return [nsr_train, nsr_test]

def random_avl(n=5000):
    avl = pd.read_csv('./data/raw/avl_set/avila-tr.txt', header=None)
    avl.columns = avl.columns.astype(str)
    avl = avl.rename(columns={'10': 'target'})
    avl_train = avl.sample(n=n, replace=False)
    avl_test = avl.drop(avl_train.index)
    avl_train.to_csv('./data/train/avl.csv', index=False)
    avl_test .to_csv('./data/test/avl.csv', index=False)
    avl_train = pd.read_csv('./data/train/avl.csv')
    avl_test = pd.read_csv('./data/train/avl.csv')

    return [avl_train, avl_test]
    
