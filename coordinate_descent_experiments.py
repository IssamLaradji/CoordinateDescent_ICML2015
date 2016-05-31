from sklearn import datasets
from coordinate_descent import CDRegressor, CDClassifier
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import pylab as plt
import time
from scipy.io import savemat, loadmat
import scipy
from matplotlib import colors

#lines = ["-", "--", "-","-.", "-"]
#linewidths = [4, 2, 2]
#colors = ['black', 'green','green']
exp_titles = {'exp1':"$\ell_2$ -regularized sparse least squares",
              'exp2':"$\ell_2$ -regularized sparse logistic regression",
              'exp3':'Over-determined dense least squares',
              'exp4':'$\ell_1$ -regularized underdetermined sparse least squares'  }


ll__ = '_logistic_lipschitz_update'
aa__ = '_logistic_line_search'
legend_names = {'cyclic': 'Cyclic',
         'random': 'Random',
         'lipschitz_sampling': 'Lipschitz',
         'GS': 'GS',
         'GSL': 'GSL',
         'GS_True': 'Approximated-GS',
         'GSL_True': 'Approximated-GSL',
         'GS_False': 'GS',
         'GSL_False': 'GSL',

         'cyclic_line_search': 'Cyclic-exact',
         'random_line_search': 'Random-exact',
         'lipschitz_sampling_line_search': 'Lipschitz-exact',
         'GS_line_search': 'GS-exact',
         'GSL_line_search': 'GSL-exact',

         'cyclic_step_update': 'Cyclic-constant',
         'random_step_update': 'Random-constant',
         'lipschitz_sampling_step_update': 'Lipschitz-constant',
         'GS_step_update': 'GS-constant',
         'GSL_step_update' : 'GSL-constant',

         'proximal_with_max': 'GSL',
         'proximal_without_max': 'GSL_without_max',
        
         'GS-q':  'GS-q',
         'GS-s': 'GS-s',
         'GS-r': 'GS-r',

         'GSL-q':  'GSL-q',
         'GSL-s': 'GSL-s',
         'GSL-r': 'GSL-r'
         }

colors_ = colors.cnames
my_colors = ['red', 'green', 'blue', 'purple', 'saddlebrown', 'darkkhaki', 'black',
             'orange']
jjj = 0
hex_ = [colors_[color] for color in my_colors]

rgb = [colors.hex2color(color) for color in hex_]
legend_colors = {'cyclic': np.array(rgb[jjj]),
         'random': np.array(rgb[jjj+1]),
         'lipschitz_sampling': np.array(rgb[jjj+2]),
         'GS': np.array(rgb[jjj+3]),
         'GSL': np.array(rgb[jjj+4]),
         'GS_True': np.array(rgb[jjj+5]),
         'GSL_True': np.array(rgb[jjj+6]),

         'GS_False': np.array(rgb[jjj+3]),
         'GSL_False':  np.array(rgb[jjj+4]),

         'cyclic_step_update': np.array(rgb[jjj]),
         'random_step_update': np.array(rgb[jjj+1]),
         'lipschitz_sampling_step_update': np.array(rgb[jjj+2]),
         'GS_step_update': np.array(rgb[jjj+3]),
         'GSL_step_update': np.array(rgb[jjj+4]),

         'cyclic_line_search': np.array(rgb[jjj+0]),
         'random_line_search': np.array(rgb[jjj+1]),
         'lipschitz_sampling_line_search': np.array(rgb[jjj+2]),
         'GS_line_search': np.array(rgb[jjj+3]),
         'GSL_line_search': np.array(rgb[jjj+4]),

         'GS-q':  np.array(rgb[jjj+3]),
         'GS-s': np.array(rgb[jjj+3]),
         'GS-r': np.array(rgb[jjj+3]),

         'GSL-q':  np.array(rgb[jjj+4]),
         'GSL-s': np.array(rgb[jjj+1]),
         'GSL-r': np.array(rgb[jjj+7])}



def plot_loss_values(fig, loss_values, name):
  """Plot figures for the loss values"""
  # Plot loss values
  x_axis = np.arange(loss_values.shape[0])
  y_axis = loss_values

  fig.set_xlabel('Epochs', fontsize=15)
  fig.set_ylabel("Objective", fontsize=15)
 
  fig.plot(x_axis, y_axis / y_axis[0], label=legend_names[name], 
                                       color=legend_colors[name])

selections_algs = ['cyclic', 'random','lipschitz_sampling', 'greedy_search', \
                   'greedy_search_lipschitz']
selections_algs = ['proximal']
fig, axis  = plt.subplots(nrows=1, ncols=1)

random_state = 5
iteration=0


exp = 'exp1'

data = loadmat("data/"+ exp + ".mat")

X, y = data['X'], data['y']

n_samples, n_features = X.shape

max_epochs = 101

if exp == 'exp1':
    assert X.shape == (1000, 1000)
    """sparse_least_square"""
    n_experiments = 5

    fVals = np.zeros(n_experiments).astype('object')
    fEvals = np.zeros(n_experiments).astype('object')
    my_color_list  = np.zeros((n_experiments, 3))
    names_mat = np.zeros(n_experiments).astype('object')
    
    #X = scipy.sparse.csc_matrix(X)
    for selection_algorithm in ['cyclic', 'random', 'lipschitz_sampling', 'GS',
                                'GSL']:
        clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                          random_state=random_state,\
                          update_algorithm='step_update',\
                          max_epochs=max_epochs,lambda_l2=1./n_samples,
                          lambda_l1=0, sanity_check=True)
        clf.fit(X, y)

        print selection_algorithm, 'completed!'

        # for matlab
        fVals[iteration] = np.arange(max_epochs).astype('float64')[:, np.newaxis]
        my_color_list[iteration] = legend_colors[selection_algorithm]
        names_mat[iteration] = legend_names[selection_algorithm]
        fEvals[iteration] = clf.loss_values[:, np.newaxis] / clf.loss_values[0]

        iteration+=1

        plot_loss_values(axis, clf.loss_values, selection_algorithm)

elif exp == 'exp2':
    """line_search_vs_1_div_L"""
    assert X.shape == (1000, 1000)
    n_experiments = 10

    fVals = np.zeros(n_experiments).astype('object')
    fEvals = np.zeros(n_experiments).astype('object')
    lineStyles = np.zeros(n_experiments).astype('object')
    my_color_list  = np.zeros((n_experiments, 3))
    names_mat = np.zeros(n_experiments).astype('object')


    for selection_algorithm in ['cyclic', 'lipschitz_sampling', 'random', 'GS', 'GSL']:
        for update_algorithm in ['step_update','line_search']:
            
            clf = CDClassifier(verbose=False,selection_algorithm=selection_algorithm, 
                                    random_state=random_state,\
                                    update_algorithm=update_algorithm,\
                                    max_epochs=max_epochs, lambda_l2=1./n_samples,
                                    sanity_check=False)
            clf.fit(X, y)
            name = selection_algorithm + '_' + update_algorithm
            plot_loss_values(axis, clf.loss_values, name)

            # for matlab
            fVals[iteration] = np.arange(max_epochs).astype('float64')[:, np.newaxis]
            my_color_list[iteration] = legend_colors[name]
            names_mat[iteration] = legend_names[name]
            fEvals[iteration] = clf.loss_values[:, np.newaxis] / clf.loss_values[0]

            if update_algorithm == 'step_update':
                lineStyles[iteration] = '-'
            else:
                lineStyles[iteration] = '--'

            iteration+=1


elif exp == 'exp3':
    """sparse_least_square"""
    assert X.shape == (1000, 100)

    n_experiments = 6

    fVals = np.zeros(n_experiments).astype('object')
    fEvals = np.zeros(n_experiments).astype('object')
    my_color_list  = np.zeros((n_experiments, 3))
    names_mat = np.zeros(n_experiments).astype('object')

    

    for selection_algorithm in ['lipschitz_sampling', 'cyclic','random']:
        clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                                random_state=random_state,lambda_l2=0,lambda_l1=0,
                                max_epochs=max_epochs)
        clf.fit(X, y)
        name = selection_algorithm
        plot_loss_values(axis, clf.loss_values, name)

        # for matlab
        my_color_list[iteration] = legend_colors[selection_algorithm]
        names_mat[iteration] = legend_names[selection_algorithm]
        fEvals[iteration] = clf.loss_values[:, np.newaxis] / clf.loss_values[0]
        fVals[iteration] = np.arange(max_epochs).astype('float64')[:, np.newaxis]

        iteration+=1


    for selection_algorithm in ['GS', 'GSL']:
        for fast_approximation in [False, True]:
            if selection_algorithm == 'GSL' and not fast_approximation:
                continue
            clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                                    random_state=random_state,lambda_l2=0,lambda_l1=0,
                                    max_epochs=max_epochs, fast_approximation=fast_approximation)
            clf.fit(X, y)
            name = selection_algorithm + '_' + str(fast_approximation)
            plot_loss_values(axis, clf.loss_values, name)

            # for matlab
            my_color_list[iteration] = legend_colors[name]
            names_mat[iteration] = legend_names[name]
            fEvals[iteration] = clf.loss_values[:, np.newaxis] / clf.loss_values[0]
            fVals[iteration] = np.arange(max_epochs).astype('float64')[:, np.newaxis]

            iteration+=1
        

#least_square_gradient(X, y, theta_prime, alpha=0, coordinate=coordinate)
elif exp == 'exp4':
    """approximated GS"""
    assert X.shape == (1000, 10000)
    #n_features = 500
   
    results = None

    n_experiments = 8

    fVals = np.zeros(n_experiments).astype('object')
    fEvals = np.zeros(n_experiments).astype('object')
    my_color_list  = np.zeros((n_experiments, 3))
    names_mat = np.zeros(n_experiments).astype('object')

    selections_algs = ['random', 'cyclic', 'lipschitz_sampling', 
                       'GS-q', 'GS-r','GS-s', 'GSL-q','GSL-r']
    for selection_algorithm in selections_algs:


        clf = CDRegressor(verbose=False,selection_algorithm=selection_algorithm, 
                          random_state=random_state,
                          update_algorithm='closed_form',\
                          max_epochs=max_epochs, lambda_l2=0,
                          lambda_l1=1./n_samples,
                          sanity_check=False)
        clf.fit(X, y)

        print selection_algorithm, 'completed!'

        plot_loss_values(axis, clf.loss_values, selection_algorithm)

        # for matlab
        my_color_list[iteration] = legend_colors[selection_algorithm]
        names_mat[iteration] = legend_names[selection_algorithm]
        fEvals[iteration] = clf.loss_values[:, np.newaxis] / clf.loss_values[0]
        fVals[iteration] = np.arange(max_epochs).astype('float64')[:, np.newaxis]

        iteration+=1

show = True
plt.legend(loc='best')

if show:
    plt.show()
else:
    plt.title(exp_titles[exp])
    plt.savefig("pythonFigures\\" + exp + ".png")
    plt.close()

    loc = 'prettyPlots\\'
    if exp == 'exp2':
        savemat(loc + str(exp) + '.mat', {'names':names_mat[:, np.newaxis] , 
            'fVals':fEvals[:, np.newaxis], 'fEvals':fVals[:, np.newaxis],
            'maxIter':100, 'colors':my_color_list,'the_title':exp_titles[exp],
            'lineStyles':lineStyles})
    else:
        savemat(loc + str(exp) + '.mat', {'names':names_mat[:, np.newaxis] , 
            'fVals':fEvals[:, np.newaxis], 'fEvals':fVals[:, np.newaxis],
            'maxIter':100, 'colors':my_color_list,'the_title':exp_titles[exp]})
