import numpy as np
from sklearn.metrics import average_precision_score
import ComplexityEvaluator
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
a=average_precision_score(y_true, y_scores)
print(a)