# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def compute(function, x, A, b, args, coordinate=None):
  L1 = args["L1"]
   
  if function == "loss":
    reg = L1 * np.sum(np.abs(x)) 

    if "b_pred" not in args:
      b_pred = safe_sparse_dot(A, x)
    else:
      b_pred = args["b_pred"]

    loss = np.sum((b - b_pred) ** 2) / 2 + reg

    return loss

  elif function == "gradient":
    if "b_pred" not in args:
      b_pred = safe_sparse_dot(A, x)
    else:
      b_pred = args["b_pred"]
    
    loss = b_pred - b

    if coordinate is None:
      grad = safe_sparse_dot(A.T, loss)
    else:
      grad = safe_sparse_dot(A[:, coordinate], loss)

    return grad

  elif function == "proximal_step":
    L = args["prox_lipschitz"]
    g_func = args["g_func"]
    L1 = args["L1"]

    g = g_func(x, A, b, args, coordinate)

    if coordinate is None:

      x_half = x - g / L

      # soft thresholding
      x = np.sign(x_half) * np.maximum(0, np.abs(x_half) - L1 / L)

    else:
      L = args["prox_lipschitz"][coordinate]
      x_half = x[coordinate] - g / L

      # soft thresholding
      x[coordinate] = np.sign(x_half) * np.maximum(0, np.abs(x_half) - L1 / L)

    return x

  elif function == "lipschitz":
    lipschitz_values = np.sum(A ** 2, axis=0)

    return lipschitz_values



def init_objective(A, b, args=None):
  if args == None:
    args = {}

  args["n_params"] = A.shape[1]
  x = np.zeros(args["n_params"])

  f_func = lambda x, A, b, args : compute("loss", x, A, b, args)
  g_func = lambda x, A, b, args, coordinate : compute("gradient", x, A, b, args, coordinate)
  ps_func = lambda x, A, b, args, coordinate : compute("proximal_step", x, A, b, args, coordinate)
  h_func = lambda x, A, b, args, coordinate : compute("hessian", x, A, b, args, coordinate)
  l_func = lambda x, A, b, args : compute("lipschitz", x, A, b, args)

  args["f_func"] = f_func
  args["g_func"] = g_func
  args["h_func"] = h_func
  args["ps_func"] = ps_func

  args["lipschitz"] = l_func(x, A, b, args)

  args["ylabel"] = "Binary logistic loss"
  
  return args


def get_feature_block(block, args):  
  return block