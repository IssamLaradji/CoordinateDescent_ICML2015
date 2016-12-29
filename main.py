import dataset_utils as du
import numpy as np
import sys
import argparse
import pandas as pd
import os
from coordinate_descent import CDPredictor

from objective_functions import lasso, logistic, least_squares

from pretty_plot import pretty_plot
from itertools import product
import json


obj_classes = {"lasso": lasso, "logistic":logistic,
               "least_squares":least_squares}


def save_csv(path, csv_file):
    create_dirs(path)
    csv_file.to_csv(path + ".csv", index=False) 

    print "csv file saved in %s" % (path)


def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default ="exp2")  
    parser.add_argument('-o','--objective', default ="logistic")

    parser.add_argument('-s','--selection_rules', nargs='+', default =["random"])
    parser.add_argument('-u','--update_rules',nargs='+', default=["quadraticLip"])
    parser.add_argument('-n','--n_iters', default=10, type=int)
    parser.add_argument('-fig','--show_fig', type=int, default=1)
    parser.add_argument('-e','--experiment', default=None)
    parser.add_argument('-L1','--L1', type=float, default=0)
    parser.add_argument('-L2','--L2', type=float, default=0)
    parser.add_argument('-t','--timeit', type=int, default=0)
    parser.add_argument('-v','--verbose', type=int, default=0)
    parser.add_argument('-title','--title', default=None)

    io_args = parser.parse_args()
    
    dataset_name = io_args.dataset

    objective = io_args.objective
    selection_rules = io_args.selection_rules        
    update_rules = io_args.update_rules
    exp = io_args.experiment
    L1 = io_args.L1
    L2 = io_args.L2
    n_iters = io_args.n_iters + 1
    verbose = io_args.verbose
    timeit = io_args.timeit
    title = io_args.title

    ##############################
    np.random.seed(0)

    title = title
    ylabel = objective
    xlabel = None

    ### 0. GET ALGORITHM PAIRS
    if exp != None:
        # LOAD EXPERIMENTS
        with open('experiments.json') as data_file:
            exp_dict = json.loads(data_file.read())

        info = exp_dict[exp]
        selection_rules = info["s_rules"]
        update_rules = info["u_rules"]
        dataset_name = info["dataset_name"]
        objective = info["objective"]
        n_iters = info["n_iters"] + 1
        L1 = info["L1"]
        L2 = info["L2"]
        title = info["Title"]
        ylabel = info["ylabel"]
        xlabel =info["xlabel"]

    # 1. Load Dataset
    dataset = du.load_dataset(dataset_name)
    A = dataset["A"]
    b = dataset["b"]
    
    n_uRules = len(update_rules)
    n_sRules = len(selection_rules)

    
    results = pd.DataFrame()
    timeResults = {}

    for s_rule, u_rule in product(selection_rules, update_rules):
        np.random.seed(1)

        clf = CDPredictor(selection_rule=s_rule, update_rule=u_rule, L2=L2, L1=L1,
                          objective=objective, 
                          verbose=verbose, timeit=timeit,
                          n_iters=n_iters)
        clf.fit(A, b)

        
        if n_uRules == 1:
            name = "%s" % (s_rule)
        elif n_sRules == 1:
            name = "%s" % (u_rule)
        else:
            name = "%s_%s" % (s_rule, u_rule)


        results[name] = clf.loss_scores
        if timeit:
            timeResults[name] = clf.time_elapsed

    if timeit:
        # Compute time result
        pp = pretty_plot.PrettyPlot(title=title, ylabel=ylabel, xlabel="Time (seconds)")

        y_array = np.array(results)
        
        labels = results.columns
        n_points, n_labels = results.shape
        x_array = np.zeros((n_points, n_labels))

        bk = np.min(timeResults.values())

        y_list = []
        x_list = []
        for j in range(n_labels):
            time_per_iter = timeResults[labels[j]] / n_points
            x_vector = np.ones(n_points) * time_per_iter
            x_vector[0] = 0.

            x_vector = np.cumsum(x_vector) 
            y_vector  = y_array[:, j]
            valid = x_vector <= bk
            x_vector = x_vector[valid]
            y_vector = y_vector[valid]

            x_list += [x_vector]
            y_list += [y_vector] 

        pp.plot(y_list, x_list, labels)
        pp.show()

    else:
        pp = pretty_plot.PrettyPlot(title=title, ylabel=ylabel, xlabel=xlabel)
        pp.plot_DataFrame(results / results.max().max())
        pp.show()

    if exp is not None:
        fpath = ("experiments/%s" % (exp))

        create_dirs(fpath)
        save_csv(fpath, results)
        pp.save(fpath)

    else:
        pp.show()