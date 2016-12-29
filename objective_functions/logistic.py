# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def compute(function, x, A, b, args, coordinate=None):
  np.testing.assert_equal(np.unique(b), np.array([-1, 1]))
  L2 = args["L2"]
   
  if function == "loss":
    reg = 0.5 * L2 * np.sum(x ** 2) 

    if "b_pred" not in args:
      b_pred = safe_sparse_dot(A, x)
    else:
      b_pred = args["b_pred"]

    loss = np.sum(np.log(1 + np.exp(- b*b_pred))) + reg

    return loss

  elif function == "gradient":
    if "b_pred" not in args:
      b_pred = safe_sparse_dot(A, x)
    else:
      b_pred = args["b_pred"]

    residual = - b / (1. + np.exp(b * b_pred))

    if coordinate is None:
      grad = safe_sparse_dot(A.T, residual)
      grad += L2 * x
    else:
      grad = safe_sparse_dot(A[:, coordinate].T, residual)
      grad += (L2 *  x[coordinate])

    return grad

  elif function == "hessian":
    if "b_pred" not in args:
      b_pred = safe_sparse_dot(A, x)
    else:
      b_pred = args["b_pred"]

    sig = 1. / (1. + np.exp(- b * b_pred))

    if coordinate is None:
      hessian = A.T.dot(np.diag(sig * (1-sig)).dot(A))
      hessian += L2
    else:
      hessian = A[:, coordinate].T.dot(np.diag(sig * \
                  (1-sig)).dot(A[:, coordinate]))
      hessian += L2

    return hessian


  elif function == "lipschitz":
    lipschitz_values = 0.25 * np.sum(A ** 2, axis=0) + L2

    return lipschitz_values

def update_residual(x, A, b, args, coordinate, x_oldCoordinate):
    # 3. FAST FORWARD PASS
    args["b_pred"] -= A[:, coordinate] * x_oldCoordinate
    args["b_pred"] += A[:, coordinate] * x[coordinate]

    #args["residual"] = b - args["b_pred"]

    return args

def init_objective(A, b, args=None):
  if args == None:
    args = {}

  if "L2" not in args:
    args["L2"] = 1.0


  args["n_params"] = A.shape[1]

  f_func = lambda x, A, b, args : compute("loss", x, A, b, args)
  g_func = lambda x, A, b, args, block : compute("gradient", x, A, b, args, block)
  h_func = lambda x, A, b, args, block : compute("hessian", x, A, b, args, block)
  l_func = lambda A, b, args : compute("lipschitz", None, A, b, args)

  args["f_func"] = f_func
  args["g_func"] = g_func
  args["h_func"] = h_func

  r_func = update_residual
  args["r_func"] = r_func


  args["lipschitz"] = l_func(A, b, args)

  args["ylabel"] = "Binary logistic loss"
  
  return args
