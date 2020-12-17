from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *

classifier = tree.DecisionTreeClassifier(max_depth=4)  # limit depth of tree
iris = load_iris()
classifier.fit(iris.data, iris.target)

viz = dtreeviz(classifier,
               iris.data,
               iris.target,
               target_name='variety',
               feature_names=iris.feature_names,
               # need class_names for classifier
               class_names=["setosa", "versicolor", "virginica"]
               )

viz.view()
