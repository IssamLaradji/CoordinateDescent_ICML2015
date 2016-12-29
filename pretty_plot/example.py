import pretty_plot
import numpy as np
import pylab as plt

if __name__ == "__main__":
    x1 = np.arange(1000)
    x2 = np.arange(1000) ** 1.2
    x3 = np.arange(1000) ** 1.3

    X = np.stack([x1, x2, x3], axis=1)
    X = - X.astype(float)
    
    pp = pretty_plot.PrettyPlot() 
    pp.plot(X)
    pp.show()







# Regression training score:  0.999999798676
# Classification training score:  0.903
# [ 0.08848751  0.21543845  0.10152939]
