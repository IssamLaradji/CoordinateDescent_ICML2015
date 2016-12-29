import pylab as plt
import matplotlib
from itertools import product
import numpy as np

import pandas as pd
import os
import plot_utils as pu
matplotlib.style.use('bmh')

markers = [("-", "o"), ("-", "p"), ("-", "D"), ("-", "^"), ("-", "s"),
               ("-", "8"), ("-", "o")]
colors = ['#741111', "#000000", '#3a49ba','#7634c9', 
          "#4C9950", "#CC29A3", '#ba3a3a', "#0f7265"]

bright_colors = ["#00C5CD"]

def setup_fig(title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=(13, 10))
    #fig = plt.figure()
    ax = fig.add_subplot(111)

    if title is not None:
        #ax.set_title(title, fontsize=14)
        fig.suptitle(title, fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)

    #fig.patch.set_facecolor('white')

    ax.set_axis_bgcolor('white')
    return fig, ax

class PrettyPlot:
    def __init__(self, title=None, ylabel=None, xlabel=None):
        fig, ax = setup_fig(title, ylabel, xlabel)

        self.fig = fig 
        self.ax = ax
        self.lim_set = False

    def show(self):
        plt.show()

    def save(self, path):
        pu.create_dirs(path)
        self.fig.savefig(path + ".png")
        print "Figure saved in %s" % (path)

    def plot_DataFrame(self, results):
        n_points, n_labels = results.shape
        
        x_vals = np.arange(n_points)
        labels = results.columns
        y_array = np.array(results)
        y_list = []
        x_list = []
        for j in range(n_labels):
            x_list += [x_vals]
            y_list += [y_array[:, j]]
        

        self.plot(y_list, x_list, labels)

    def set_lim(self, ylim, xlim):
        self.lim_set = True
        self.ax.set_ylim(ylim) 
        self.ax.set_xlim(xlim)   
              
    def plot(self, y_list, x_list, labels=None):
        fig = self.fig
        ax = self.ax 

        label_positions, label_indices = pu.get_labelPositions(y_list, x_list)

        ls_markers = markers
        n_labels = len(y_list)

        if labels is None:
            labels = map(str, np.arange(n_labels))
        
        lw = 2.5

        if not self.lim_set:
            y_min, y_max = pu.get_min_max(y_list)
            x_min, x_max = pu.get_min_max(x_list)

            ax.set_ylim([y_min, y_max]) 
            ax.set_xlim([x_min, x_max]) 

        for i in range(n_labels):
            color = colors[i]
            ls, marker = ls_markers[i]

            y_vals = y_list[i]
            x_vals = x_list[i]

            n_points = len(y_vals)

            label = labels[i]

            if i > 0:
                percentage = pu.get_overlapPercentage(i, y_list)
                if percentage > 0.3:
                    ls = "--"
                    color = bright_colors[0]

            markerFreq = n_points / (int(np.log(n_points)) + 1)
            line, = ax.plot(x_vals, y_vals, markevery=markerFreq, 
                    markersize=8, color=color, lw=lw, alpha=0.9,
                    label=label, ls=ls, marker=marker)

            x_point, y_point = label_positions[i]
            angle = pu.get_label_angle(x_vals, y_vals, label_indices[i], ax, color='0.5', size=12)

            ax.text(x_point , y_point, label, va='center',ha='center', 
                    rotation=angle,
                    color=color, 
                    bbox=pu.box_color(color, ls, marker), 
                    fontsize=14)

        return fig, ax

def plot_csv(results, fig, ax):   

    for rank, column in enumerate(results.columns):
        color = colors[2*rank]
        ls, marker = markers[rank]
        n_points = results.shape[0]

        freq = n_points / (int(np.log(n_points)) + 1)
        ax.plot(results[column], markevery=freq, 
                markersize=8,
                color=color, lw=2.5, label=column, ls=ls, marker=marker)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, prop={'size':10},
               ncol=2, mode="expand", borderaxespad=0.,fancybox=True, shadow=True)

    plt.tight_layout(pad=7)


    return fig, ax
