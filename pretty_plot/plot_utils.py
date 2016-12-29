import pylab as plt
import matplotlib
from itertools import product
import numpy as np

import pandas as pd
import os


def get_overlapPercentage(index, y_list):
    n_points = 0
    for i in range(index + 1):
        n_points = min(n_points, y_list[i].size)

    y_vector = y_list[index][:n_points, np.newaxis]

    prev_lines = np.zeros((n_points, index))

    for i in range(index):
        prev_lines[:, i] = y_list[i][:n_points]
        prev_lines /=  np.linalg.norm(prev_lines, axis=0)

    y_norm = y_vector / np.linalg.norm(y_vector, axis=0)

    diff = np.abs((prev_lines - y_norm)).min(axis=1)

    n_overlap = np.sum(diff < 1e-6)
    percentage = n_overlap / float(n_points)

    return percentage

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass  


# LABEL POSITIONS
def get_labelPositions(y_list, x_list):
    """Get label positions greedily"""
    n_labels = len(y_list)

    # GET BORDER POINTS
    x_min, x_max = get_min_max(x_list)
    x_mid = (x_max - x_min) / 2

    y_min, y_max = get_min_max(y_list)
    y_mid = (y_max - y_min) / 2
    # Border points
    bp1 = np.array(list(product([x_min, x_max, x_mid], 
                                [y_min, y_max, y_mid])))[:-1]

    # Top right points
    # bp2 = np.array(list(product([0., 1.0, 0.75], 
    #                             [0., 1.0, 0.75])))[:-1]

    # Bottom right points
    # bp3 = np.array(list(product([0., 1.0, 0.25], 
    #                             [0., 1.0, 0.25])))[:-1]    
    #border_points = np.vstack([bp1, bp2, bp3])
    border_points = np.vstack([bp1])
    n_border = border_points.shape[0]

    # Initialize placeholders
    ref_points = np.zeros((n_border + n_labels, 2))

    label_positions = np.zeros((n_labels, 2))
    label_indices = np.zeros(n_labels, int)

   
    
    ref_points[:n_border] = border_points

    for i in range(n_labels):
        # GET POSITIONS
        n_points = x_list[i].size
        xy_points = np.zeros((n_points, 2))

        xy_points[:, 0] = x_list[i]
        xy_points[:, 1] = y_list[i]
   
        # GET REF POINTS
        dist = get_pairwise_distances(xy_points, ref_points[:n_border + i])

        # GET MINIMUM DISTANCES
        min_dist = dist.min(axis=1)

        # GET MAXIMUM MINIMUM DISTANCE
        label_index = np.argmax(min_dist)
        label_pos = xy_points[label_index]

        ref_points[n_border + i] = label_pos
        label_positions[i] = label_pos
        label_indices[i] = label_index

    return label_positions, label_indices

def get_min_max(v_list):
    vector = v_list[0]
    v_min = np.min(vector)
    v_max = np.max(vector)

    for i in range(1, len(v_list)):
        vector = v_list[i]
        v_min = min(np.min(vector), v_min)
        v_max = max(np.max(vector), v_max)

    return v_min, v_max

def get_label_angle(xdata, ydata, index, ax, color='0.5', size=12, window=3):
    n_points = xdata.size

    x1 = xdata[index]  
    y1 = ydata[index]

    #ax = line.get_axes()

    sp1 = ax.transData.transform_point((x1, y1))

    slope_degrees = 0.
    count= 0.

    for i in range(index+1, min(index+window, n_points)):     
        y2 = ydata[i]
        x2 = xdata[i]

        sp2 = ax.transData.transform_point((x2, y2))

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        slope_degrees += np.degrees(np.arctan2(rise, run))
        count += 1.

    for i in range(index-1, max(index-window, 0), -1):
        y2 = ydata[i]
        x2 = xdata[i]

        sp2 = ax.transData.transform_point((x2, y2))

        rise =  - (sp2[1] - sp1[1])
        run = -(sp2[0] - sp1[0])

        slope_degrees += np.degrees(np.arctan2(rise, run))
        count += 1.

    slope_degrees /= count

    return slope_degrees


def box_color(edgecolor, linestyle, marker):
    """Creates box shape"""
    return dict(facecolor="white",
                edgecolor=edgecolor, linestyle=linestyle,
                #hatch=marker,
                linewidth=2, boxstyle="round")

def get_pairwise_distances(A, B):
    # GET EUCLIDEAN DISTANCES
    n_A = A.shape[0]
    n_B = B.shape[0]

    A_square = np.dot(A ** 2, np.ones((2, n_B)))
    B_square = np.dot(B ** 2, np.ones((2, n_A)))

    AB = np.dot(A, B.T)

    dist = A_square + B_square.T - 2 * AB

    return np.sqrt(dist + 1e-10)
