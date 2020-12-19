# COGS118A Final Project

Shuyuan Wang
A 14534701

## Abstract

     Supervised machine learning models come the one of the most efficient ways solving everyday problem. In this project, I will shows how to use ['ACC', 'PRC','FSC', 'LFT', 'ROC', 'APR', 'RMS', 'MXE'] as basic approaches to evaluate models. Later, by calculating MEAN (mean of above scores]) and 'OPT-SEL'(the one produced after looking at the test scores), we will get the better understand to the overall performance and show in t test graphs.

## Introduction

    In the era of big data, applications of Machine Learning and Data Science have penetrated all walks of life. Therefore, Data Science has evolved as one of the most promising majors for students who expect to keep pace with the times and wish to have the ability to make a difference in this era. It's not unfamiliar when talks about machine learning or big data, but how exactly we could establish a effecient machine learning models on the big data sets?

    There  always a "fight" between supervised and unsupervised machine learning, people keep evaluating them through different aspects. Today, I would more like to give my own defination to unsupervised machine learning rather than evaluating effeciencies of these two types of models, which might be largely depends on data or ones' professionals. For me, Supervised machine learning would more like math problem. One could calculate them by hands through the functions behinds every methodolog if the data size is relatively small. Even one couls based on those functions to write codes since the theory would rarely changed. You could make a plan and write down in the paper to give a struction about the sequence accomplishing each steps. With strong mathmatical background, one could give a relatively accurate hypothesis.


## Method

### Data
- AVILA
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * TARGET: {Class: A, B, C, D, E, F, G, H, I, W, X, Y} [Multilabels]

- ADULT
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * TARGET: >50K, <=50K. [Binarylabels]

- NURSERY 
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * TARGET: NURSERY colunm [Multilabels]

- Individual household electric power consumption Data Set [ EXTRA_WORKS ]
    * GOAL: Predict the future trand of individual household electric power consumprion.
    * TARGET: voltage: minute-averaged voltage (in volt)
    * Theme: Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.
    * Data Set Characteristics:[Time series; Multivariables]
    * TASK: [Regression, Clustering]

### scorings :

    The DataFrame: 
    group1: ACC, FSC, LFT
    group2: ROC, APR, BEP
    group3: RMS, MXE
    group4: OPT-SELS

    Within the Matric: 
    - The algorithm with the best performance on each metric is ***boldfaced***. Using t-test p= 0.05, others are *'ed in three trails if they still have good performance.

- The *threshold* metrics are accuracy(ACC), F-score(FSC) and lift(LFT).
- The *rank metrics* we use are area under the ROC curve (ROC),average precision(APR), and precision/recall break even point (BEP).
- The *probability metrics*,squared error(RMS) and cross-entropy (MXE),interpret the predicted value of each case as the conditional probability of that case being in the positive class.
- The last column,*OPT-SEL*,is the mean normalized score for the eight metrics when model selection is done by cheating and looking at the final testsets.
    
    
### Algorithm ()

1. *KNN*: 
knn_params = {'classifier': [KNeighborsClassifier()],
              'classifier__n_neighbors': np.random.randint(1, 50, 10),
              'classifier__weights': ['uniform', 'distance'],
              'classifier__algorithm': ['brute']} 

2. *Decisiontrees(DT)*: 

dtr_params = {'classifier': [DecisionTreeClassifier()],
              'classifier__min_samples_split':  range(2, 403, 10),
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_depth': [2, 4, 6, 8, 'log2'],
              'classifier__strategy': ['mean', 'median']}
3. *SVMs*:
C = [0.005, 0.001, 0.01, 0.05, 0.5,1, 2 ]
#factors of ten from 10-7 to 103
gamma = [10**(-5), 10, 1, 0, 0.01]
svm_params = {'classifier': [SVC()],
              'classifier__C': C,
              'classifier__gamma': gamma,
              'classifier__kernel': ['rbf', 'poly'],
              }


### Trails
    - 3 dataset
    - 3 algorithms
    - 7 scores
    3*3*7 trails in total

    In order to reduce the running time, I recorded every models locally following certain pattern of paths. There are two main folders storing my models so that I make these two big folders, path=['all_models/', 'best_models/'] to seperatly store clf.pkl and clf.best_estimator_.pkl, then based on the classidier, I established three roots. Since every dataset will exicuted through such same process, I created three dataset colder to contain scores under each classifier. 
    For later usage, I could follow the path best_classifier -> model -> data -> score to get pkl file.


        

    
### Time complexity 

Here are the basic understanding of time complexity
    - Good:
        * O(1) => Constant Time
        * O(log n) => Logarithmic Time
        * O(√n) => Square Root Time
    - OK: 
        * O(n) => Linear Time
    - Bad:
        * O(n log n) => Linearithmic Time
    - Awful:
        * O(n^2) => Quadratic Time / Polynomial Time
        * O(2^n) => Exponential Time
        * O(n!) => Factorial Time

    In addition to showing the time learning graph, I refered package online which could help me to analyze the time complexity of each trails.
## Experiment
    The biggest trouble would be errors I meet during training model. How to carefully design write my own scorer, balance between y, score and classifier makes my whole structure became clear and more robotic. Since I chose two of my datsets are multivariable as well as multiclasses, I need to add other restrictions on existing scorers or I need to write my own ones, such as ROC APR. I got trouble training nsr-svm-ROC and nsr-svm-APR, especially the last trail, those factors never conpromise to each other if I used general setting. Firstly, I set y value to be "on_hot coded" by LabelBinarizer(). Based the mathmatical functions, getting y_prob and y_pred from such transformed y set, useing these two number to generate score from soc_score. While for APR, I carefully followed the instruction about how it could be cauculated by hands and write coded on my own.

### Data Processing
1. adult: there are missing value "?" in their object columns and extreme values as as 0 and 999 in their int/float columns. I excluded them first and then find out that there are two columns that are not sufficient enough. In addition, from the descripton of the dataset, as the author that the " " are only based on personal expectation rather than actual fact, which would not provide any help to our dataset. After data analyzing, I found our that "education_num" and "education" has same contribution to the final prediciton. Thus, I dropped one of them to reduce redundency. All columns I dropped from this dataset are ['education','fnlwgt','capital-gain','capital-loss']. 
2. avl: This is the dataset that fully composed by fload. 
3. nursery: Firstly, it's very easy to see that the columns are all binarized or ordinal. I assignered specific order from lowest to highest value of such variable. Then I assigned integer only to the label in otder to avoid value_error caused by ROC score. While when I trained this dataset, MXE makes me realize that this dataset is unbalanced so that I used replace to reduce this outlier. 

## Results and Discussion
### Combo- 3 trails

### Combo - 3*3 trails

### Best best_estimator_

### Hyperparameters
These *heatmap-style plots* showing the relationship between  *validation performance* vs *hyperparameter setting* for different algorithms

### Learning evaluation
*learning curves* could help to compare test set performances for the
best hyperparameter choice while we trained such model using different number of training dataset. 



•Written clarity –Refer to the figure table; Explain what it says; Corroborate the hypothesis;
I used quite a amount of time balancing the y dimensions. Since my datasets are not binarized classified, dealing wuth multiclass and multifeature needs me to put lots of time on it. Especially the trouble I got facing SVM+ROC combo.  It's worth to try on every possibles solving combos like this but I used Labelbinarizer to make y became [[0010],[1000],[0100],[1110]] and then transpose them back to 0 and 1 in one dimension.

## Conclusion and Future Work

### Conclusion
    By using GridsearchCV, I got my best hyperparameters as well as score for each dataset- classifier combo. While from the chars as well as learning graphs above, we could get better conclision if there are more epoches for each algorithms. 

### Future workssss
    - For now, although I established functions enabling to do less hard code, they certainly could be approved by adding exceptions or include more cases. Once such functions could be used as a package, there will be keywords such as  "AUTO" "GridSearchCV - hyperparameter" "auto model selection".
    - In addition, the data preprocessing might need to be improved to consider details details based on different data sets. For most of scores in the paper, they only support binay labled, which is quite impossible solving real problem. In my project, I only did labelBinarizer to y for saving time. But I believe there must be other ways worth to try. 
## Reference

https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/

https://towardsdatascience.com/supervised-learning-algorithms-explanaition-and-simple-code-4fbd1276f8aa

https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568)

https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html

https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn