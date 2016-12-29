import numpy as np 

import numpy as np

from sklearn.datasets import make_regression, make_classification
from scipy.io import savemat, loadmat
 
def load_dataset(name):

    if name == "very_sparse":
        # l2- regularized sparse least squares
        data = loadmat("datasets/very_sparse.mat")
        A, b = data['A'].toarray(), data['b']

        b = b.ravel()

    if name == "binary_sparse":
        # l2- regularized sparse least squares
        data = loadmat("datasets/very_sparse.mat")
        A, b = data['A'].toarray(), data['b']

        b = b.ravel()
        n_samples = A.shape[0]
        block = np.random.choice(np.arange(n_samples), size=n_samples/2, replace=False)

        non_block = np.setdiff1d(np.arange(n_samples), block)
        b[block] = 1
        b[non_block] = -1

    if name == "exp1":
        # l2- regularized sparse least squares
        data = loadmat("datasets/exp1.mat")
        A, b = data['X'], data['y']
        b = b.ravel()
        
    elif name == "exp2":
        # l2- regularized sparse logistic regression
        data = loadmat("datasets/exp2.mat")
        A, b = data['X'], data['y']
        b = b.ravel()
   
    elif name == "exp3":
        # Over-determined dense least squares
        data = loadmat("datasets/exp3.mat")
        A, b = data['X'], data['y']
        b = b.ravel()
        
    elif name == "exp4":
        # L1 - regularized underdetermined sparse least squares
        data = loadmat("datasets/exp4.mat")
        A, b = data['X'], data['y']
        b = b.ravel()


    return {"A":A,"b": b}


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y