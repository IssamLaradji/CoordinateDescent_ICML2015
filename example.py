from sklearn import datasets
from coordinate_descent import CDRegressor, CDClassifier

import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    n_samples, n_features = 1000, 500

    # Regression
    X, y = datasets.make_regression(n_samples, n_features)

    reg = CDRegressor(selection_rule='GSL', n_iters=20, verbose=True)

    # Train regressor
    reg.fit(X, y)

    print "Regression training score: ", reg.score(X,y)

    # Classification
    X, y = datasets.make_classification(n_samples, n_features)
    y[y==0] = -1
    clf = CDClassifier(selection_rule='GSL', n_iters=20)

    # Train classifier
    clf.fit(X, y)

    print "Classification training score: ", clf.score(X,y)

# Regression training score:  0.999999798676
# Classification training score:  0.903
# [ 0.08848751  0.21543845  0.10152939]
