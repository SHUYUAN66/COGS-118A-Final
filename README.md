# CODE PART

replicate  a part of the analysis done in the paper by Caruana &Niculescu-Mizil (hereafter referred to as CNM06).

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

FOR EXTRA CREDIT
2. ANN
3. automl

## Experiment Trails  

•Randomly choose *5000 samples* from the datasets
•Each trial
    - 5 fold CV

    - For possible combinations of *hyper-parameter settings*.
        (reference: https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568)
        * Grid Search
            https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
        * Random Search
        * Bayesian Optimisation
•Code + Report to be submitted.


## Results:

Precision, Recall, MCC, accuracy_score,roc_curve, confusion_matrix, classification_report

Accuracy(Acc.) and time (t) result for cancer, diabetes and banknote datasets
respectively


## EXTRA CREDIT!! 

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

