import numpy as np
import pylab as pl
from itertools import cycle
from sklearn.utils.extmath import safe_sparse_dot
import time
import numpy as np
import update_rules as ur
import selection_rules as sr
from sklearn.utils import check_random_state
from sklearn.utils.fixes import expit as logistic_sigmoid
from sklearn.base import RegressorMixin
from sklearn.utils.fixes import expit as logistic_sigmoid
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.neighbors import NearestNeighbors
from objective_functions import logistic, least_squares, lasso
from numba_utils import heapOps

LOSS_FUNCTIONS = {"logistic": logistic, "lasso":lasso,
                  "least_squares":least_squares}

class BaseCoordinateDescent(BaseEstimator):
  def __init__(self, n_iters=25, objective='least_squares', selection_rule='GS', 
               update_rule='quadraticLip', L2=0, 
               nearest_neighbor=False, L1=0, 
               random_state=None, verbose=False, intercept=False, 
               timeit=False):

    self.n_iters = n_iters
    self.L2 = L2
    self.selection_rule = selection_rule
    self.nearest_neighbor = nearest_neighbor
    self.update_rule = update_rule
    self.verbose = verbose
    self.random_state = random_state
    self.L1 = L1
    self.objective = objective
    self.intercept = intercept
    self.l_func = LOSS_FUNCTIONS[objective]
    self.random_state = random_state
    self.timeit = timeit

    if timeit:
      self.verbose = False

    rng = check_random_state(self.random_state)

    self.args = {"L2":L2, "L1":L1, "rng":rng}

  def fit(self, A, b):
    """Train the coordinate descent algorithm"""

    # Add intercept
    if self.intercept:
      A = np.hstack([np.ones((A.shape[0], 1)), A])


    self.args = self.l_func.init_objective(A, b, self.args)
    self.x = np.zeros(self.args["n_params"])

    n_samples, n_features = A.shape
    
    # This outputs a warning when a 1d array is expected
    if b.ndim == 2 and b.shape[1] == 1:
        b = column_or_1d(b, warn=False)

    # SET UP LOSS, SELECTION AND UPDATE FUNCTIONS
    f_func = self.args["f_func"]
    r_func = lambda x, args, coordinate, x_oldCoordinate : self.args["r_func"](x, A, b, args, coordinate, x_oldCoordinate)
    s_func = lambda x, args: sr.select_coordinate(self.selection_rule, x, A, b, args)
    u_func = lambda x, args, coordinate: ur.update_coordinate(self.update_rule,
                                                              x, A, b, args, coordinate)

    g_func = lambda x, args, coordinate: self.args["g_func"](x, A, 
                                                             b, args, coordinate)
    # Run Algorithm
    self.args["b_pred"] = safe_sparse_dot(A, self.x)

    coordinate = -1

    self.loss_scores = np.zeros(self.n_iters)

    time_flag = True

    for epoch in range(self.n_iters):
      if time_flag and epoch > 0 and self.timeit:
        s_time = time.clock()
        time_flag = False

      self.args["epoch"] = epoch

      # Compute loss
      loss = f_func(self.x, A, b, self.args)  
      self.loss_scores[epoch] = loss
      
      if self.verbose:
          print_string = "%d - coordinate %d - Loss %.3f" % (epoch, coordinate, loss)
          print "%s\n%s" % (print_string, "-" * len(print_string))

      #1. SELECT COORDINATE #######################
      coordinate, self.args = s_func(self.x, self.args)

      # keep track of the old coordinate for the forward pass
      x_oldCoordinate = self.x[coordinate]

      #2. UPDATE COORDINATE #######################
      self.x, self.args = u_func(self.x, self.args, coordinate)

      # 3. FAST FORWARD PASS - Compute Residue
      self.args = r_func(self.x, self.args, coordinate, x_oldCoordinate)

      # 4. UPDATE HEAP
      if self.selection_rule in ['heapGSL', "heapGS"]:
        self.args = heapOps.update_maxGradient(coordinate, self.x, self.args, g_func, self.selection_rule)

    if self.timeit:
      e_time = time.clock()
      self.time_elapsed = e_time - s_time
      print "%s - time_elapse: %.3f second" % (self.selection_rule,
                                               self.time_elapsed)

    if self.verbose:
      print_string = "s_rule: %s\nu_rule: %s" % (self.selection_rule, self.update_rule)
      print "\n%s\n%s" % (print_string, "=" * len(print_string))


    # Plot results

  def _decision_scores(self, A):
    """Predict"""
    n_samples, n_features = A.shape;
    # Add intercept
    if self.intercept:
      A = np.hstack([np.ones((n_samples,1)), A])

    return A.dot(self.x)

class CDPredictor(BaseCoordinateDescent):
  def __init__(self, n_iters=25, objective="least_squares", selection_rule='GS',
               update_rule='quadraticLip', L2=0,
               nearest_neighbor=False, L1=0, 
               random_state=None, verbose=False, timeit=False):
    super(CDPredictor, self).__init__(n_iters=n_iters,
                                        objective=objective,
                                       selection_rule=selection_rule,
                                       update_rule=update_rule,
                                       L2=L2,
                                       L1=L1,
                                       verbose=verbose,
                                       nearest_neighbor=nearest_neighbor,
                                       random_state=random_state,
                                       timeit=timeit)
  def predict(self, A):
    b_scores = self._decision_scores(A)

    return b_scores

class CDClassifier(BaseCoordinateDescent, ClassifierMixin):
  def __init__(self, n_iters=25, selection_rule='GS',
               update_rule='quadraticLip', L2=0,
               nearest_neighbor=False, L1=0, 
               random_state=None, verbose=False):
    super(CDClassifier, self).__init__(n_iters=n_iters,
                                       objective="logistic",
                                       selection_rule=selection_rule,
                                       update_rule=update_rule,
                                       L2=L2,
                                       L1=L1,
                                       verbose=verbose,
                                       nearest_neighbor=nearest_neighbor,
                                       random_state=random_state)

  def predict(self, X):
    y_scores = logistic_sigmoid(self._decision_scores(X))
    
    y_scores[y_scores >= 0.5] = 1
    y_scores[y_scores < 0.5] = -1

    return y_scores

class CDRegressor(BaseCoordinateDescent, RegressorMixin):
  def __init__(self, n_iters=25, selection_rule='GS',
               update_rule='quadraticLip', L2=0,
               nearest_neighbor=False, L1=0, 
               random_state=None, verbose=False):
    super(CDRegressor, self).__init__(n_iters=n_iters,
                                       objective="least_squares",
                                       selection_rule=selection_rule,
                                       update_rule=update_rule,
                                       L2=L2,
                                       L1=L1,
                                       verbose=verbose,
                                       nearest_neighbor=nearest_neighbor,
                                       random_state=random_state)
  def predict(self, X):
    y_scores = self._decision_scores(X)

    return y_scores