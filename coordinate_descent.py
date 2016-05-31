import numpy as np
import pylab as pl
from itertools import cycle
from sklearn.utils.extmath import safe_sparse_dot
import time
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.fixes import expit as logistic_sigmoid
from sklearn.base import RegressorMixin
from sklearn.utils.fixes import expit as logistic_sigmoid
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.neighbors import NearestNeighbors

#### Gradient Functions
def least_square_gradient(X, y, theta, alpha=0, y_pred=None, coordinate=None):
    """Compute the gradient for each feature."""
    if y_pred is None:
      y_pred = safe_sparse_dot(X, theta)
    
    loss = y_pred - y

    if coordinate is None:
      grad = safe_sparse_dot(X.T, loss)
      grad += alpha * theta
    else:
      grad = safe_sparse_dot(X[:, coordinate], loss)
      grad += (alpha *  theta[coordinate])

    return grad


def logistic_gradient(X, y, theta, alpha=0, y_pred=None, coordinate=None):
    """Compute the gradient for each feature."""
    if y_pred is None:
      y_pred = safe_sparse_dot(X, theta)

    if coordinate is None:
      grad = -(X.T.dot(y / (1. + np.exp(y * y_pred))))
      grad += alpha * theta
    else:
      grad = - X[:, coordinate].T.dot(y / (1 + np.exp(y * y_pred)))
      grad += (alpha *  theta[coordinate])

    return grad

def logistic_hessian(X, y, theta, alpha=0, y_pred=None, coordinate=None):
    """Compute the hessian for each feature."""
    if y_pred is None:
      y_pred = safe_sparse_dot(X, theta)

    sig = 1. / (1. + np.exp(- y * y_pred))

    if coordinate is None:
      hessian = X.T.dot(np.diag(sig * (1-sig)).dot(X))
      hessian += alpha
    else:
      hessian = X[:, coordinate].T.dot(np.diag(sig * \
                  (1-sig)).dot(X[:, coordinate]))
      hessian += alpha

    return hessian


#### Loss functions
def squared_loss(X, y, theta, alpha=0, y_pred=None):
    """Compute the square error."""
    reg = 0.5 * alpha * np.sum(theta ** 2)

    if y_pred is None:
      y_pred = X.dot(theta)

    return ((y - y_pred) ** 2).sum() / 2  + reg

def log_loss(X, y, theta, alpha=0, y_pred=None):
    reg = 0.5 * alpha * np.sum(theta ** 2)

    if y_pred is None:
      y_pred = X.dot(theta)

    loss = np.sum(np.log(1+np.exp(- y * y_pred))) + reg

    return loss

#### UPDATE FUNCTIONS

def least_square_step_update(X, y, theta, alpha=0, coordinate=None, \
                                  y_pred=None, lipschitz_values=None):
    """Update the weight w as w = w - (1/L) * grad_j, where L = \sum_i aij^2"""
    single_gradient = least_square_gradient(X, y, theta, alpha=alpha,\
                      y_pred=y_pred, coordinate=coordinate)

    return theta[coordinate] - (1. / lipschitz_values[coordinate]) * single_gradient

def logistic_step_update(X, y, theta, alpha=0, coordinate=None, \
                                  y_pred=None, lipschitz_values=None):
    """Update the weight w as w = w - (1/L) * grad_j, where L = \sum_i aij^2"""
    single_gradient = least_square_gradient(X, y, theta, alpha=alpha,\
                      y_pred=y_pred, coordinate=coordinate)

    return theta[coordinate] - (1. / lipschitz_values[coordinate]) * single_gradient

def gradient_test(self, X, y, gradient_function):

    if gradient_function == least_square_gradient:
      loss_function = squared_loss

    elif gradient_function == logistic_gradient:
      loss_function = log_loss

    grad = gradient_function(X, y, self.theta, alpha=self.lambda_l2)
    e = np.zeros(X.shape[1])

    eps = 2 * np.sqrt(1e-12) * (1 + np.linalg.norm(X))
    col = 1
    e[col] = eps
    g_up = loss_function(X, y, self.theta + e)
    g_down = loss_function(X, y, self.theta - e)
    
    np.testing.assert_almost_equal((g_up - g_down)/(2 * eps), grad[col], 3)
    print 'gradient test passed!'

class BaseCoordinateDescent(BaseEstimator):
  def __init__(self, max_epochs=25, loss='squared_loss', selection_algorithm='GS', \
               update_algorithm='closed_form', lambda_l2=0, \
               fast_approximation=False, lambda_l1=0, random_state=None, verbose=False,
               intercept=True, sanity_check=False):
    self.max_epochs = max_epochs
    self.lambda_l2 = lambda_l2
    self.selection_algorithm = selection_algorithm
    self.theta = None
    self.fast_approximation = fast_approximation
    self.update_algorithm = update_algorithm
    self.verbose = verbose
    self.random_state = random_state
    self.lambda_l1 = lambda_l1
    self.loss = loss
    self.intercept = intercept
    self.sanity_check = sanity_check

    self.theta = None

  def _compute_distances(self, X, y, lipschitz_values, gradient_function, 
                         y_pred):
    """Compute the distance between w_t+1 and w_t"""
    if self.lambda_l1 == 0:
      learning_rate = 1. / lipschitz_values
      gradient = gradient_function(X, y, self.theta, alpha=self.lambda_l2,
                                   y_pred=y_pred)

      return - learning_rate * gradient

    else:
      theta_prime = self._proximal_gradient(X, y, lipschitz_values, \
                                                y_pred, gradient_function)
      
      return theta_prime - self.theta
    

  def _proximal_gradient(self, X, y, lipschitz_values, y_pred, gradient_function,
                         coordinate=None):
    # Thesis chapter 3 equation 
    learning_rate = 1. / lipschitz_values
    gradient = gradient_function(X, y, self.theta, alpha=self.lambda_l2, \
                                   y_pred=y_pred, coordinate=coordinate)

    theta_half_prime = self.theta - learning_rate * gradient

    L1_value = np.abs(theta_half_prime) - self.lambda_l1 / lipschitz_values
    L1_value = L1_value.clip(0.)

    theta_prime = np.sign(theta_half_prime) * L1_value

    ####
    if coordinate is None:
      return theta_prime
    else:
      return theta_prime[coordinate]


  def fit(self, X, y):
    """Train the coordinate descent algorithm"""
    # Add intercept
    if self.intercept:
      X = np.hstack([np.ones((X.shape[0], 1)), X])

    n_samples, n_features = X.shape
    
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                     dtype=np.float64, order="C", multi_output=True)

    # This outputs a warning when a 1d array is expected
    if y.ndim == 2 and y.shape[1] == 1:
        y = column_or_1d(y, warn=False)

    rng = check_random_state(self.random_state)
    #init = np.sqrt(6. / n_features)
    #self.theta = rng.uniform(-init, init, n_features)
    if self.theta is None:
      self.theta = np.zeros(n_features)
      #self.theta = rng.uniform(-1, 1, n_features) * 0.5 - 0.5
    # Determine whether its classification or regression
    if isinstance(self, RegressorMixin):
      # its regression - least square problem
      loss_function = squared_loss
      gradient_function = least_square_gradient
      update_function = least_square_step_update

      lipschitz_values = np.sum(X ** 2, axis=0) + self.lambda_l2

    else:
      # its binary classification - logistic optimization problem
      loss_function = log_loss
      gradient_function = logistic_gradient
      update_function = logistic_step_update

      lipschitz_values = 0.25 * np.sum(X ** 2, axis=0) + self.lambda_l2

    # please remove later
    self.L = lipschitz_values
    ## Some set up
    if self.selection_algorithm == 'lipschitz_sampling':
      lipschitz_cum_sum = np.cumsum(lipschitz_values / 
                                    np.sum(lipschitz_values))
    
    if self.sanity_check:
      gradient_test(self, X, y, gradient_function)

    ####### KNN
    # Create tree for approximate greedy
    if self.selection_algorithm == 'GSL' and self.fast_approximation:
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean', 
                           algorithm='ball_tree')
        Extended_X = np.hstack([X, -X])
        L = np.sum(Extended_X ** 2, axis=0).ravel()

        r = Extended_X / np.sqrt(L)

        #for j in range(n_features):
            #print np.linalg.norm(r[:,j])
        knn.fit(r.T)
        print 'Approximated_GSL here'

    elif self.selection_algorithm == 'GS' and self.fast_approximation:
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean', 
                           algorithm='ball_tree')
        knn.fit(np.hstack([X, -X]).T)
        print 'Approximated_GS here'

    #######
    self.loss_values = np.zeros(self.max_epochs)
    self.coordinates = np.zeros(self.max_epochs)
    self.gradients = np.zeros((self.max_epochs, n_features))
    self.theta_values = np.zeros((self.max_epochs, n_features))

    # Run Algorithm
    y_pred = safe_sparse_dot(X, self.theta)

    for epoch in range(self.max_epochs):
        
        # Compute global loss
        global_loss = loss_function(X=X, y=y, theta=self.theta, y_pred=y_pred,
                                    alpha=self.lambda_l2) 
        # Add L1 Norm
        global_loss += self.lambda_l1 * np.sum(abs(self.theta))
        
        self.loss_values[epoch] = global_loss
        self.theta_values[epoch] = self.theta
        #########################
        # Compute which coordinate to update
        if self.selection_algorithm == 'cyclic':
          # Get the next coordinate
          coordinate = epoch % n_features

        elif self.selection_algorithm == 'random':
          # Get a random coordinate from a uniform distribution
          coordinate = rng.randint(n_features)

        elif self.selection_algorithm == 'lipschitz_sampling':
          # Get a random coordinate from the lipschitz distribution
          value = rng.uniform(0, 1)
             
          coordinate = np.searchsorted(lipschitz_cum_sum, value)

        elif self.selection_algorithm == 'GS':
          # Standard maximization of inner product search
          if not self.fast_approximation:
            grad_list = gradient_function(X, y, self.theta, self.lambda_l2, y_pred=y_pred)
            ### Please remove later
            self.gradients[epoch] = np.abs(grad_list)
            ########
            coordinate = np.argmax(np.abs(grad_list))
            
          else:
            loss = y_pred - y
            coordinate = knn.kneighbors(loss)[1][0][0]

            coordinate %= n_features
          
        elif self.selection_algorithm == 'GSL':
          if not self.fast_approximation:
            # Standard maximization of inner product search
            grad_list = gradient_function(X, y, self.theta, self.lambda_l2, y_pred=y_pred)
            grad_list /= np.sqrt(lipschitz_values)

            ### Please remove later
            self.gradients[epoch] = np.abs(grad_list)
            ########
            coordinate = np.argmax(np.abs(grad_list))
          else:
            loss = y_pred - y
            coordinate = knn.kneighbors(loss)[1][0][0]

            coordinate %= n_features

        elif (self.selection_algorithm == 'GSL-q' or 
              self.selection_algorithm == 'GS-q'):

          if self.selection_algorithm == 'GSL-q':
            # Each coordinate has its own lipschitz value
            lipschitz_prime = lipschitz_values

          elif self.selection_algorithm == 'GS-q':
            # Max of lipschitz values
            lipschitz_prime = np.ones(lipschitz_values.size) * \
                              np.max(lipschitz_values)
          # w_{i+1} - w_i = - 1/L_i nabla f(wt)
          distances = self._compute_distances(X, y, lipschitz_values=lipschitz_prime, 
                                              gradient_function=gradient_function, 
                                              y_pred=y_pred)

          L2_term_grad_list = gradient_function(X, y, self.theta, self.lambda_l2, y_pred=y_pred)

          # Incorporating the L1 term values
          L1_term_distances = self.lambda_l1 * abs(distances + self.theta)
          L1_term_list =  self.lambda_l1 * abs(self.theta)

          # Complicated Equation in page 7 of the paper
          values = L2_term_grad_list * distances + (lipschitz_prime/2.) \
                   * distances * distances + L1_term_distances - L1_term_list

          #print('f_nabla: %f, d: %f, g_d: %f, g: %f' ) % (f_nabla, distance, g_d, g)

          # Sanity check (make sure no value is positive)
          if self.sanity_check:
             assert np.sum(values > 1e-8) == 0
             print 'values are positive test passed!'

          ### Please remove later
          self.gradients[epoch] = values
          ########
          coordinate = np.argmin(values)

        elif (self.selection_algorithm == 'GSL-r' or
              self.selection_algorithm == 'GS-r'):

          if self.selection_algorithm == 'GSL-r':
            # Each coordinate has its own lipschitz value
            lipschitz_prime = lipschitz_values

          elif self.selection_algorithm == 'GS-r':
            # Max of lipschitz values
            lipschitz_prime = np.ones(lipschitz_values.size) * \
                              np.max(lipschitz_values)
          
          # w_{i+1} - w_i = - 1/L_i nabla f(wt)
          distances = self._compute_distances(X, y, 
                                              lipschitz_values=lipschitz_prime,
                                              gradient_function=gradient_function, 
                                              y_pred=y_pred)
          
          ### Please remove later
          self.gradients[epoch] = np.abs(distances)
          ########
          coordinate = np.argmax(np.abs(distances))

        elif  (self.selection_algorithm == 'GSL-s' or
              self.selection_algorithm == 'GS-s'):

          grad_list = gradient_function(X, y, self.theta, self.lambda_l2, 
                                              y_pred=y_pred)

          grad_list_prime = np.zeros(grad_list.shape)
          # Point zero-valued variables in the right direction
          ind_neg = grad_list < -self.lambda_l1
          ind_pos = grad_list > self.lambda_l1

          grad_list_prime[ind_neg] = grad_list[ind_neg] + self.lambda_l1 
          grad_list_prime[ind_pos] = grad_list[ind_pos] - self.lambda_l1 

          # Compute the real gradient for non zero-valued variables
          non_zero_indices = abs(self.theta) > 1e-3
          grad_list_prime[non_zero_indices] = grad_list[non_zero_indices] + \
                                              self.lambda_l1 * \
                                              np.sign(self.theta[non_zero_indices])
          
          if self.selection_algorithm == 'GSL-s':
            grad_list_prime /= np.sqrt(lipschitz_values)

          ### Please remove later
          self.gradients[epoch] = np.abs(grad_list_prime)
          ########
          coordinate = np.argmax(abs(grad_list_prime))

        ### Update theta
        self.coordinates[epoch] = coordinate
        # Fast update of y_pred
        old_theta_coordinate = self.theta[coordinate]
        
        if self.update_algorithm == 'step_update':
          single_gradient = gradient_function(X, y, self.theta, alpha=self.lambda_l2,\
                                              y_pred=y_pred, coordinate=coordinate)

          learning_rate = 1. / lipschitz_values[coordinate]
          self.theta[coordinate] -= learning_rate * single_gradient
          
          #### sanity check
          # Gradient must be zero at new point
          if self.sanity_check:
            # Gradient should be almost zero at new point
            new_grad = gradient_function(X, y, self.theta, alpha=self.lambda_l2, 
                                             coordinate=coordinate)
            #assert abs(new_grad) < 1e-6
            #print 'zero grad test passed!'

        elif self.update_algorithm == 'closed_form':
          self.theta[coordinate] = self._proximal_gradient(X, y, lipschitz_values, y_pred, gradient_function,
                                                          coordinate=coordinate)
          if self.sanity_check:
            # Gradient should be almost zero at new point
            new_grad = gradient_function(X, y, self.theta, alpha=self.lambda_l2, 
                                             coordinate=coordinate)

            #assert abs(new_grad) < 1e-6
            #print 'zero grad test passed!'

        elif self.update_algorithm == 'line_search':
          # Run line search
          lower_bound, upper_bound = -1000000., 1000000.
          e = np.zeros(X.shape[1])
          e[coordinate] = 1
          tmp_y_pred = y_pred.copy()

          # Start with lipschitz value
          step_size = 1./lipschitz_values[coordinate]

          for i in range(200): 
              tmp_y_pred = y_pred - X[:, coordinate] * self.theta[coordinate]
              tmp_y_pred += X[:, coordinate] * (self.theta + step_size * e)[coordinate]
              
              grad_new = gradient_function(X, y, self.theta + step_size * e,
                                                 y_pred=tmp_y_pred,
                                                 coordinate=coordinate,
                                                 alpha=self.lambda_l2) 
              if abs(grad_new) < 1e-10:
                break

              if grad_new > 0:
                  upper_bound = step_size
              elif grad_new < 0:
                  lower_bound = step_size
              
              step_size = np.random.uniform(lower_bound, upper_bound, size=None)

          self.theta[coordinate] += step_size
        # Change one element from y_pred
        y_pred -= X[:, coordinate] * old_theta_coordinate
        y_pred += X[:, coordinate] * self.theta[coordinate]
        #print global_loss
        # break
        #if global_loss < 1000000:
        #  print 'broken'
        #  break
        # sanity check - loss should always decrease
        #if epoch != 0:
        #  print self.update_algorithm
          #assert global_loss <= self.loss_values[epoch - 1] + 1e-3
        # Print progress
        if self.verbose:
            print "----------------------------"
            print "Epoch", epoch, " Loss value :", global_loss

  def _decision_scores(self, X):
    """Predict"""
    n_samples, n_features = X.shape;
    # Add intercept
    X = np.hstack([np.ones((n_samples,1)), X])

    return X.dot(self.theta)

class CDClassifier(BaseCoordinateDescent, ClassifierMixin):
  def __init__(self, max_epochs=25, selection_algorithm='GS',
               update_algorithm='closed_form', lambda_l2=0,
               fast_approximation=False, lambda_l1=0, 
               random_state=None, verbose=False, sanity_check=False):
    super(CDClassifier, self).__init__(max_epochs=max_epochs,
                                       selection_algorithm=selection_algorithm,
                                       update_algorithm=update_algorithm,
                                       lambda_l2=lambda_l2,
                                       lambda_l1=lambda_l1,
                                       verbose=verbose,
                                       sanity_check=sanity_check,
                                       random_state=random_state)
    
  def fit(self, X, y):
    classes = np.unique(y)

    if classes.size != 2:
      raise ValueError("Algorithm only supports binary classification")

    self.pos_class = classes[1]
    self.neg_class = classes[0]

    y_new = y.copy()

    y_new[y == self.pos_class] = 1
    y_new[y == self.neg_class] = -1

    super(CDClassifier, self).fit(X, y_new)

    return self

  def predict(self, X):
    y_scores = logistic_sigmoid(self._decision_scores(X))
    
    y_scores[y_scores >= 0.5] = self.pos_class
    y_scores[y_scores < 0.5] = self.neg_class

    return y_scores

class CDRegressor(BaseCoordinateDescent, RegressorMixin):
  def __init__(self, max_epochs=25, selection_algorithm='GS',
               update_algorithm='closed_form', lambda_l2=0,
               fast_approximation=False, lambda_l1=0, 
               random_state=None, verbose=False, sanity_check=False):
    super(CDRegressor, self).__init__(max_epochs=max_epochs,
                                       selection_algorithm=selection_algorithm,
                                       update_algorithm=update_algorithm,
                                       lambda_l2=lambda_l2,
                                       lambda_l1=lambda_l1,
                                       verbose=verbose,
                                       sanity_check=sanity_check,
                                       fast_approximation=fast_approximation,
                                       random_state=random_state)
  def predict(self, X):
    y_scores = self._decision_scores(X)

    return y_scores
