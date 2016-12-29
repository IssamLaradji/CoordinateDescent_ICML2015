# -*- coding: utf-8 -*-
import numpy as np
from numba_utils import heapOps
from sklearn.utils.extmath import safe_sparse_dot

from sklearn.neighbors import NearestNeighbors
                 
def select_coordinate(rule, x, A, b, args):
  """ Adaptive selection rules """
  
  n_params = x.size
  n_samples, n_features = A.shape

    
  g_func = args["g_func"]

  epoch = args["epoch"]
  lipschitz = args["lipschitz"]
  rng = args["rng"]

  if rule == 'cyclic':
      # Get the next coordinate
      coordinate = epoch % n_params

  elif rule == 'random':
      # Get a random coordinate from a uniform distribution
      coordinate = rng.randint(n_params)

  elif rule in ['heapGSL', "heapGS"]:
      # use heap to keep track of gradients
      if epoch == 0:
        if rule == "heapGS":
          scores = np.abs(g_func(x, A, b, args, None))

        elif rule == "heapGSL":
          g = g_func(x, A, b, args, None)
          gL = g / np.sqrt(lipschitz)
          scores = np.abs(gL)

        heap, h2l, l2h, n_elements = heapOps.create_heap(scores)
        d_matrix, d_index, max_depend = get_dependancy_matrix(A)

        args["d_matrix"] = d_matrix
        args["d_index"] = d_index
        args["heap"] = heap
        args["h2l"] = h2l
        args["l2h"] = l2h
        args["n_elements"] = n_elements
      
      h2l = args["h2l"]
      l2h = args["l2h"]
      
      return args["h2l"][0], args

  elif rule == 'lipschitz':

    if epoch == 0:
      L = args["lipschitz"]
      args["lipschitz_cumSum"] = np.cumsum(L / np.sum(L))
    
    value = rng.uniform(0, 1)
    coordinate = np.searchsorted(args["lipschitz_cumSum"], value)

  elif rule == 'GS':
    g = g_func(x, A, b, args, None)
    coordinate = np.argmax(np.abs(g))

  elif rule == 'GSL':
    lipschitz = args["lipschitz"]

    g = g_func(x, A, b, args, None)
    gL = g / np.sqrt(lipschitz)

    coordinate = np.argmax(np.abs(gL))

  elif rule in ["GSL-q", "GS-q"]:
    L = args["lipschitz"]
    L1 = args["L1"]

    if rule == "GS-q":
      Lprime = np.ones(n_params) * np.max(L)

    elif rule == "GSL-q":
      Lprime = L
     #block = np.unravel_index(block,  (n_features, n_classes))
    args["prox_lipschitz"] = Lprime
    dist = compute_dist(x, A, b, args)

    g = g_func(x, A, b, args, None)

    # Incorporating the L1 term values
    L1_next = L1 * abs(dist + x)
    L1_current =  L1* abs(x)
    
    # Complicated Equation in page 7 of the paper
    scores = g * dist + (Lprime / 2.) * dist * dist + L1_next - L1_current

    coordinate = np.argmin(scores)

  elif rule in ["GSL-r", "GS-r"]:
    L = args["lipschitz"]

    if rule == "GS-r":
      Lprime = np.ones(n_params) * np.max(L)

    elif rule == "GSL-r":
      Lprime = L
     #block = np.unravel_index(block,  (n_features, n_classes))
    args["prox_lipschitz"] = Lprime
    dist = compute_dist(x, A, b, args)

    # Complicated Equation in page 7 of the paper
    coordinate = np.argmax(np.abs(dist))

  elif rule in ["GSL-s", "GS-s"]:
    L1 = args["L1"]

    g = g_func(x, A, b, args, None)

    gP = np.zeros(g.shape)
    # Point zero-valued variables in the right direction
    ind_neg = g < - L1
    ind_pos = g > L1

    gP[ind_neg] = g[ind_neg] + L1
    gP[ind_pos] = g[ind_pos] - L1

    # Compute the real gradient for non zero-valued variables
    nonZeros = abs(x) > 1e-3
    gP[nonZeros] = g[nonZeros] + L1 * np.sign(x[nonZeros])
    
    if rule == 'GSL-s':
      gP /= np.sqrt(L)

    coordinate = np.argmax(abs(gP))

  elif rule in ["approxGS", "approxGSL"]:
    lipschitz = args["lipschitz"]
    if args["epoch"] == 0:
      knn = NearestNeighbors(n_neighbors=1, metric='euclidean', 
                             algorithm='ball_tree')
      if rule == "approxGS":
        A_scale = A
      elif rule == "approxGSL":
        A_scale = A / np.sqrt(lipschitz)

      A_ext = np.hstack([A_scale, -A_scale])

      args["knn"] = knn.fit(A_ext.T)

    r = args["b_pred"] - b
    coordinate = args["knn"].kneighbors(r)[1][0][0]

    coordinate %= n_features


  else:
    print "selection rule %s doesn't exist" % rule
    raise

  return coordinate, args


def compute_dist(x, A, b, args):
  """Compute the distance between w_t+1 and w_t"""
  ps_func = args["ps_func"]
  g_func = args["g_func"]
  L1 = args["L1"]
  L = args["lipschitz"]

  if L1 == 0:
    g = g_func(x, A, b, args, None)

    return - g / L

  else:

    xP = ps_func(x, A, b, args, None)
    
    return xP - x


def get_dependancy_matrix(A):
  AA = safe_sparse_dot(A.T, A)
  correlation = AA > 0

  n_params = correlation.shape[0]
  n_depend = np.sum(correlation)

  max_depend = np.max(np.sum(correlation, axis=1))

  d_matrix = np.ones((n_depend, max_depend), "int32") * -1
  d_index = np.zeros(n_params + 1, "int32")

  for i in range(n_params):
      depend = np.where(correlation[i] > 0)[0]

      d_index[i] = depend.size

      d_matrix[i, :d_index[i]] = depend
  
  return d_matrix, d_index, max_depend

# def get_dependancy_matrix(A):
#   AA = safe_sparse_dot(A.T, A)
#   correlation = AA > 0

#   n_params = correlation.shape[0]
#   n_depend = np.sum(correlation)

#   max_depend = np.max(np.sum(correlation, axis=1))

#   d_matrix = np.ones(n_depend, "int32") * -1
#   d_index = np.zeros(n_params + 1, "int32")

#   for i in range(n_params):
#       depend = np.where(correlation[i] > 0)[0]

#       d_index[i + 1] = d_index[i] + depend.size

#       d_matrix[d_index[i]: d_index[i+1]] = depend
  
#   return d_matrix, d_index, max_depend
