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

### Algorithm

1. *KNN*: 

we use 26 values of K ranging fromK= 1 to K=|trainset|. We useKNN with Euclidean distance and Euclideand distance weighted by gain ratio.We also use distance weighted KNN,and locally weighted averaging.The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the trainset.

2. *Decisiontrees(DT)*: 

Decision tree is a simple but powerful learning technique that is considered as one of the famous learning algorithms that have been successfully used in practice for various classification tasks. They have the advantage of producing a comprehensible classification model with satisfactory accuracy levels in several application domains.


3. *SVMs*:

we  usethefollowingkernelsinSVMLight(Joachims,1999):linear,polynomialdegree2 & 3, radial with width {0.001,0.005,0.01,0.05,0.1,0.5,1,2}. We also vary the regularization parameter by factors of ten from 107 to 103 with each kernel.



### Evaluation scores

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
## Experiment

TODO

## Results and Discussion
### Combo- 3 trails

### Conbo - 3*3 trails

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
    By using GridsearchCV, I got my best hyperparameters as well as 

### Future workssss
    - For now, although I established functions enabling to do less hard code, they certainly could be approved by adding exceptions or include more cases. Once such functions could be used as a package, there will be keywords such as  "AUTO" "GridSearchCV - hyperparameter" "auto model selection".
    - In addition, the data preprocessing might need to be improved to consider details details based on different data sets. For most of scores in the paper, they only support binay labled, which is quite impossible solving real problem. In my project, I only did labelBinarizer to y for saving time. But I believe there must be other ways worth to try. 
## Reference

https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/

https://towardsdatascience.com/supervised-learning-algorithms-explanaition-and-simple-code-4fbd1276f8aa

https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568)

https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html

https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn