# CODE PART

replicate  a part of the analysis done in the paper by Caruana & Niculescu-Mizil (hereafter referred to as CNM06).

## Data

•3 datasets, 3 algorithms, 3 experiment trials
    From CNM06, pick 3 of the datasets (available from UCI machine learning repository) and pick 3 of the algorithms (different kernels of SVM are not 2 different classifiers, pick truly differentones). For each classifier and dataset combo, you will need to do 3 trials (CNM06 does 5; we will make it easier for you). That’s 3x3x3 = 27 total trials.

## Algorithms

1. REGULAR:
    - SVMS
    - KNN
    - Decision Tress
        - Decision tree is a simple but powerful learning technique
that is considered as one of the famous learning algorithms that have
been successfully used in practice for various classification tasks. They
have the advantage of producing a comprehensible classification model
with satisfactory accuracy levels in several application domains. 


**FOR EXTRA CREDIT (fancy things)**
2. ANN
3. automl

## Experiment Trails  

•Randomly choose *5000 samples* from the datasets
•Each trial
    - 5 fold CV

    - For possible combinations of *hyper-parameter settings*.
        (reference: https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568)
        * Grid Search (only use this)
            https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
        * Random Search
        * Bayesian Optimisation
        
•Code + Report to be submitted.

### Hyperparameters:
1. *KNN*: we use 26 values of K ranging fromK= 1 to K=|trainset|. We useKNN with Euclidean distance and Euclideand distance weighted by gain ratio.We also use distance weighted KNN,and locally weighted averaging.The kernel widths for locally weighted averaging vary from 20 to 210 times the minimum distance between any two points in the trainset.
2. *Decisiontrees(DT)*: we varythesplittingcrite-rion,pruningoptions,andsmoothing(LaplacianorBayesiansmoothing).We useallof thetreemodelsin Buntine'sINDpackage(Buntine& Caruana,1991):BAYES,ID3,CART, CART0,C4,MML,andSMML.We alsogeneratetreesof type C44LS(C4withnopruningandLaplaciansmoothing),C44BS(C44withBayesiansmoothing),andMMLLS(MMLwithLapla-ciansmoothing).See(Provost& Domingos,2003)fora descriptionof C44LS
3. *SVMs*:we  usethefollowingkernelsinSVMLight(Joachims,1999):linear,polynomialdegree2 & 3, radial with width {0.001,0.005,0.01,0.05,0.1,0.5,1,2}. We also vary the regularization parameter by factors of ten from 107 to 103 with each kernel.

## Results:

- scoring(defult = AUC) [F1, AUC]

- First Result (2 tables)
    1. a table of mean (across 3 trials) test set performance for each algorithm/dataset combo​1​.
    2. a table of mean (across 3 trials x 3 datasets) test set performance for each algorithm.
 - Second Result (3 things)
    1. A main matter table showing mean​ training​ set performance for the optimalhyperparameters on each of the dataset/algorithm combos and a discussion of thedifference between each algorithms’ training and test set performance.
    2. An appendix table with raw test set scores, not just the mean scores of the table.
    3. An appendix table with the p-values of the comparisons across algorithms in the differentmain matter tables.
    4. An analysis of the time complexity of the algorithms you tried-
    5. A learning curve per algorithm/dataset combo: comparing test set performance for thebest hyperparameter choice as you vary the number of training samples or a givendataset
    6. A heatmap-style plot of the validation performance vs hyperparameter setting for youralgorithms


## EXTRA CREDIT!! (second result 4.5.6)

Secondary results you may wish to report (extra credit land):
- An analysis of the **time complexity** of the algorithms you tried
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

    reference: 
    https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/

    https://towardsdatascience.com/supervised-learning-algorithms-explanaition-and-simple-code-4fbd1276f8aa

- A **learning curve** per algorithm/dataset combo: comparing test set performance for the
best hyperparameter choice as you vary the number of training samples or a given
dataset
- A **heatmap-style plot** of the *validation performance* vs *hyperparameter setting* for your algorithms

