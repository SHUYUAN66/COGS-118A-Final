# CODE PART

replicate  a part of the analysis done in the paper by Caruana & Niculescu-Mizil (hereafter referred to as CNM06).

## Data

•3 datasets, 3 algorithms, 3 experiment trials
    From CNM06, pick 3 of the datasets (available from UCI machine learning repository) and pick 3 of the algorithms (different kernels of SVM are not 2 different classifiers, pick truly differentones). For each classifier and dataset combo, you will need to do 3 trials (CNM06 does 5; we will make it easier for you). That’s 3x3x3 = 27 total trials.
### Dataset Information

1. ADULT
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * TARGET: >50K, <=50K.
    * COLUMNS/ Variables: ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
        - age: continuous.
        - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        - fnlwgt: continuous.
        - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        - education-num: continuous.
        - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        - sex: Female, Male.
        - capital-gain: continuous.
        - capital-loss: continuous.
        - hours-per-week: continuous.
        - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

2. AVILA
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * TARGET: {Class: A, B, C, D, E, F, G, H, I, W, X, Y}
    * VARIABLES/ ATTRIBUTES: A LIST OF STRING: 
            - F1: intercolumnar distance
            - F2: upper margin
            - F3: lower margin
            - F4: exploitation
            - F5: row number
            - F6: modular ratio
            - F7: interlinear spacing
            - F8: weight
            - F9: peak number
            - F10: modular ratio/ interlinear spacing

3. NURSERY 
    * Data Set Characteristics: [Multivariables]
    * Associated Tasks: [Classification]
    * Theme: NURSERY Evaluation of applications for nursery schools
    * TARGET: NURSERY
    * ATTRIBUTES :[EMPLOY, STRUCT_FINAN, STRUCTURE, SOC_HEALTH]
            => [parents, has_nurs, form, children, housing,finance, social, health]
    
        1. EMPLOY Employment of parents and child's nursery
            - parents Parents' occupation : usual, pretentious, great_pret
            - has_nurs Child's nursery: proper, less_proper, improper, critical, very_crit
        2. STRUCT_FINAN Family structure and financial standings
            - STRUCTURE Family structure
                - form Form of the family: complete, completed, incomplete, foster
                - children Number of children: 1, 2, 3, more
            - housing Housing conditions: convenient, less_conv, critical
            - finance Financial standing of the family : convenient, inconv
        3. SOC_HEALTH Social and health picture of the family
            - social Social conditions: non-prob, slightly_prob, problematic
            - health Health conditions: recommended, priority, not_recom

4. Individual household electric power consumption Data Set [EXTRA_CREDIT!!]
    * GOAL: Predict the future trand of individual household electric power consumprion.
    * TARGET: voltage: minute-averaged voltage (in volt)
    * Theme: Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.
    * Data Set Characteristics:[Time series; Multivariables]
    * TASK: [Regression, Clustering]
    * Notes:
        1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.
        2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.
    * ATTRIBUTES: 
        1.date: Date in format dd/mm/yyyy
        2.time: time in format hh:mm:ss
        3.global_active_power: household global minute-averaged active power (in kilowatt)
        4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
        5.global_intensity: household global minute-averaged current intensity (in ampere)
        6.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). 
            It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
        7.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). 
            It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
        8.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). 
            It corresponds to an electric water-heater and an air-conditioner


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

