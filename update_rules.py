import numpy as np

def update_coordinate(rule, x, A, b, args, coordinate):
  g_func = args["g_func"]
  lipschitz = args["lipschitz"]

  if rule == "quadraticLip":
    grad_list = g_func(x, A, b, args, coordinate)

    L = lipschitz[coordinate]
    grad_list /= L

    x[coordinate] = x[coordinate] - grad_list

    return x, args


  if rule == "proximalL1":
    ps_func = args["ps_func"]
    args["prox_lipschitz"] = args["lipschitz"]
    x = ps_func(x, A, b, args, coordinate)

    return x, args

  if rule == "exact":
    # Run line search
    lower_bound, upper_bound = -1000000., 1000000.
    e = np.zeros(A.shape[1])
    e[coordinate] = 1
    #tmp_b_pred = b_pred.copy()

    # Start with lipschitz value
    L = args["lipschitz"]
    step_size = 1. / L[coordinate]

    b_predOrig = args["b_pred"].copy()

    for i in range(200): 
      x_new = x + step_size * e

      args["b_pred"] = b_predOrig.copy()

      args["b_pred"] -= A[:, coordinate] * x[coordinate]
      args["b_pred"] += A[:, coordinate] * x_new[coordinate]

      g_new = g_func(x_new, A, b, args, coordinate) 

      if abs(g_new) < 1e-10:
        break

      if g_new > 0:
          upper_bound = step_size
      elif g_new < 0:
          lower_bound = step_size
      
      step_size = np.random.uniform(lower_bound, upper_bound, size=None)

    x[coordinate] += step_size
    args["b_pred"] = b_predOrig
    #grad = g_func(x, A, b, args, coordinate) 

    return x, args
  else:
    print "update rule %s doesn't exist" % rule
    raise
