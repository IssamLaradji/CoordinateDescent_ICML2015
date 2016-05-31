import numpy as np

from sklearn import datasets
from coordinate_descent import CDRegressor, CDClassifier

n_samples, n_features = 1000, 500

# Regression
X, y = datasets.make_regression(n_samples, n_features)

reg = CDRegressor(selection_algorithm='GSL', max_epochs=20)

# Train regressor
reg.fit(X, y)

print "Regression training score: ", reg.score(X,y)

# Classification
X, y = datasets.make_classification(n_samples, n_features)

clf = CDClassifier(selection_algorithm='GSL', max_epochs=20)

# Train classifier
clf.fit(X, y)

print "Classification training score: ", clf.score(X,y)
