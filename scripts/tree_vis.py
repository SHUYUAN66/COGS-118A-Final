import joblib


clf = joblib.load('models/knn/ROC_all.pkl')
# n_neibors, weights

print(clf.scores_)


