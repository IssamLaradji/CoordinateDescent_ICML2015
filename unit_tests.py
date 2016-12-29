from sklearn import datasets
from coordinate_descent import CDRegressor
import coordinate_descent as cd
import numpy as np
from numpy import array
from sklearn.utils.extmath import safe_sparse_dot
import pylab as plt
import time
import utilities as ut

from numpy.testing import assert_almost_equal as assert_equal
#lines = ["-", "--", "-","-.", "-"]
#linewidths = [4, 2, 2]
#colors = ['black', 'green','green']
bias = 1; 
scaling = 10; 
sparsity = 10; 
solutionSparsity = 0.1;

n_samples, n_features = 1000, 1000
X = np.random.randn(n_samples,n_features)+bias;
X = safe_sparse_dot(X, np.diag(scaling * np.random.randn(n_features)));
X = X * (np.random.rand(n_samples, n_features) < sparsity* np.log(n_samples) / n_samples);
w = np.random.randn(n_features,1) * (np.random.rand(n_features,1) < solutionSparsity);
y = X.dot(w) + np.random.randn(n_samples,1);

def test_GS_equivalency():
    selections_algs = ['GS', 'GS-r','GS-s','GS-q']

    results = None
    for selection_algorithm in selections_algs:
        clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                                    random_state=3,\
                                    update_algorithm='step_update',\
                                    max_epochs=100, lambda_l1=0)

        clf.fit(X, y)

        if results is None:
            results = clf.loss_values 
        else:
            np.testing.assert_array_equal(results, clf.loss_values)

def test_GSL_equivalency():
    selections_algs = ['GSL', 'GSL-s','GSL-q']

    results = None
    for selection_algorithm in selections_algs:
        clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                                    random_state=3,\
                                    update_algorithm='step_update',\
                                    max_epochs=100, lambda_l1=0)

        clf.fit(X, y)

        if results is None:
            results = clf.loss_values
        else:
            np.testing.assert_array_equal(results, clf.loss_values)

def test_closed_form_equal_step_update():
    update_algorithms = ['step_update', 'closed_form']

    results = None
    for update_algorithm in update_algorithms:
        clf = CDRegressor(verbose=False,selection_algorithm='GS', 
                                    random_state=3,\
                                    update_algorithm=update_algorithm,\
                                    max_epochs=100, lambda_l1=0)

        clf.fit(X, y)

        if results is None:
            results = clf.loss_values
        else:
            np.testing.assert_array_equal(results, clf.loss_values)

def test_lipschitz_constant(lipschitz_values):
    
    lipschitz_cum_sum = np.cumsum(lipschitz_values / np.sum(lipschitz_values))
    times = 1000

    ll = np.zeros(lipschitz_cum_sum.size)
    for i in range(times):
        value = np.random.uniform(0, 1)
        ll[np.searchsorted(lipschitz_cum_sum, value)]+=1
    print ll

def test_number_of_comparisons():
    #a = np.random.rand(100)
    #a.sort()
    #print 'n_samples', a.size
    b = ut.counter()

    b.binary_search(a, a[8])
    print b.n_comparisons 

def test_selections(selection_method='cyclic', lambda_l2=0, lambda_l1=1.1):
    CD = cd.CDRegressor()

    CD.theta = np.zeros(3)
    X = np.array([[2.,3,5],[7,1,2],[0,1,2]])
    y = np.array([0.5,0.2,0.4])

    loss_function = cd.squared_loss
    gradient_function = cd.least_square_gradient
    update_function = cd.least_square_step_update

    lipschitz_values = np.sum(X ** 2, axis=0) + lambda_l2

    max_epochs = 5
    loss_values = np.zeros(max_epochs)
    coordinates = np.zeros(max_epochs)

    CD.lambda_l1 = lambda_l1
    CD.lambda_l2 = lambda_l2

    n_samples, n_features = X.shape

    for epoch in range(max_epochs):
        loss_values[epoch] = loss_function(X, y, CD.theta)
        # Compute which coordinate to update
        if selection_method == 'cyclic':
            # Get the next coordinate
            coordinate = epoch % n_features

        elif selection_method == 'random':
            # Get a random coordinate from a uniform distribution
            coordinate = rng.randint(n_features)

        coordinates[epoch] = coordinate

        CD.theta[coordinate] = CD._proximal_gradient(X, y, 
                                        lipschitz_values,
                                        y_pred=safe_sparse_dot(X, CD.theta),
                                        gradient_function=gradient_function,
                                        coordinate=coordinate)
    print  'For', selection_method
    print 'Selected coordinate'
    print coordinates
    print 'Loss values'
    print loss_values

def better_test_selections(selection_algorithm='cyclic', lambda_l1=1.1):
    #### 
    CD = cd.CDRegressor(selection_algorithm=selection_algorithm,
                        max_epochs=6, lambda_l2=0, lambda_l1=lambda_l1)
    X = np.array([[2.,3],[7,1]])
    y = np.array([0.5,0.2])

    n_samples, n_features = X.shape

    if lambda_l1 == 0:
        CD.theta = np.zeros(n_features)

    else:
        CD.theta = np.array([1., -1])
    CD.fit(X, y)
    """
    print CD.L
    
    print  'For', selection_algorithm
    print 'Selected coordinates'
    print repr(CD.coordinates + 1)
    print 'Loss values'
    print repr(CD.loss_values)
    print 'Coordinate scores'
    print repr(CD.gradients)
    print 'Theta values'
    print repr(CD.theta_values)
    """
    assert_equal(CD.L, [ 53.,  10.])

    # Theta must be zeros
    if selection_algorithm == 'GS' and lambda_l1 == 0:
       #assert CD.coordinates[0] == 0
       assert_equal(CD.coordinates[:2], np.array([0,  1]))
       assert_equal(CD.loss_values[:2], np.array([0.145,  0.09066038]))
       assert_equal(CD.gradients[0], np.array([2.4,  1.7]))
       assert_equal(CD.theta_values[1:3], np.array([[ 0.04528302,  0.        ],
                                                   [ 0.04528302,  0.11113208]]))
    # Theta must be zeros
    elif selection_algorithm == 'GSL' and lambda_l1 == 0:
       #assert CD.coordinates[0] == 0
       assert_equal(CD.coordinates[:2], np.array([1,  0]))
       assert_equal(CD.loss_values[:2], np.array([0.145, 0.0005]))
       assert_equal(CD.gradients[0], np.array([0.3296654,  0.5375872]))
       assert_equal(CD.theta_values[1:3], np.array([[ 0. ,         0.17      ],
      
                                                 [ 0.00358491 , 0.17      ]]))
    # Theta must be zeros
    elif selection_algorithm == 'GS' and lambda_l1 == 1.1:
       assert_equal(CD.coordinates + 1, array([ 1.,  2.,  1.,  2.,  2.,  2.]))
       assert_equal(CD.loss_values, array([ 20.145     ,   6.01584906,   1.69836508,   0.32165792,
         0.13688412,   0.13688412]))
       assert_equal(CD.gradients, array([[ 37.6       ,   1.3       ],
       [  1.1       ,   8.19245283],
       [ 10.98018868,   1.1       ],
       [  1.1       ,   1.86306515],
       [  0.10801531,   1.1       ],
       [  0.10801531,   1.1       ]]))
       assert_equal(CD.theta_values, array([[ 1.        , -1.        ],
                   [ 0.26981132, -1.        ],
                   [ 0.26981132, -0.07075472],
                   [ 0.04188323, -0.07075472],
                   [ 0.04188323,  0.0055518 ],
                   [ 0.04188323,  0.0055518 ]]))
    elif selection_algorithm == 'GSL-s' and lambda_l1 == 1.1:
       assert_equal(CD.coordinates + 1, array([ 1.,  2.,  1.,  2.,  1.,  2.]))
       assert_equal(CD.loss_values, array([ 20.145     ,   6.01584906,   1.69836508,   0.32165792,
         0.13688412,   0.12760079]))
       assert_equal(CD.gradients, array([[  5.31585382e+00,   6.32455532e-02],
       [  4.27002410e-16,   2.93853160e+00],
       [  1.65934153e+00,   2.10650008e-16],
       [  1.83001033e-16,   9.37003472e-01],
       [  1.36259577e-01,   0.00000000e+00],
       [  3.05001722e-17,   7.69435913e-02]]))
       assert_equal(CD.theta_values, array([[ 1.        , -1.        ],
       [ 0.26981132, -1.        ],
       [ 0.26981132, -0.07075472],
       [ 0.04188323, -0.07075472],
       [ 0.04188323,  0.0055518 ],
       [ 0.02316654,  0.0055518 ]]))
    elif selection_algorithm == 'GSL-r' and lambda_l1 == 1.1:
       assert_equal(CD.gradients, array([[  7.30188679e-01,   2.00000000e-02],
       [  5.55111512e-17,   9.29245283e-01],
       [  2.27928088e-01,   5.55111512e-17],
       [  2.77555756e-17,   7.63065148e-02],
       [  1.87166923e-02,   0.00000000e+00],
       [  6.93889390e-18,   2.43317000e-02]]))

    elif selection_algorithm == 'GSL-q' and lambda_l1 == 1.1:
       assert_equal(CD.gradients, array([[ -1.41291509e+01,  -2.00000000e-03],
       [  0.00000000e+00,  -4.31748398e+00],
       [ -1.37670716e+00,   0.00000000e+00],
       [  0.00000000e+00,  -1.84773798e-01],
       [ -9.28333613e-03,   0.00000000e+00],
       [  0.00000000e+00,  -2.96015812e-03]]))



better_test_selections(selection_algorithm='GS', lambda_l1=0)
better_test_selections(selection_algorithm='GSL', lambda_l1=0)

better_test_selections(selection_algorithm='GS', lambda_l1=1.1)

better_test_selections(selection_algorithm='GS', lambda_l1=1.1)

better_test_selections(selection_algorithm='GSL-s', lambda_l1=1.1)
better_test_selections(selection_algorithm='GSL-r', lambda_l1=1.1)
better_test_selections(selection_algorithm='GSL-q', lambda_l1=1.1)

#better_test_selections(selection_algorithm='GSL', lambda_l1=1.1)
#########################
#test_selections()
#better_test_selections(selection_algorithm='cyclic')

#test_number_of_comparisons()
lipschitz_values = [0.5,0.2,0.3]
test_lipschitz_constant(lipschitz_values)
test_GS_equivalency()
test_GSL_equivalency()
test_closed_form_equal_step_update()

def approximate_GS():
    CD = cd.CDRegressor(selection_algorithm='GS',
                        fast_approximation=True)

    CD.theta = np.zeros(3)
    X = np.array([[2.,3,5],[7,1,2],[0,1,2]])
    y = np.array([0.5,0.2,0.4])

    loss_function = cd.squared_loss
    gradient_function = cd.least_square_gradient
    update_function = cd.least_square_step_update

    lipschitz_values = np.sum(X ** 2, axis=0)

    max_epochs = 5
    loss_values = np.zeros(max_epochs)
    coordinates = np.zeros(max_epochs)

    CD.lambda_l1 = 0
    CD.lambda_l2 = 0

    n_samples, n_features = X.shape
    CD.fit(X, y)


    print 'Selected coordinates'
    print repr(CD.coordinates + 1)
    print 'Loss values'
    print repr(CD.loss_values)
    print 'Coordinate scores'
    print repr(CD.gradients)
    print 'Theta values'
    print repr(CD.theta_values)


#approximate_GS()
