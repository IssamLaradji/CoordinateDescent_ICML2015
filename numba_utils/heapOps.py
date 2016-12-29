import numpy as np
from numba import autojit
import time
from numba import float64, int32

@autojit("float64[:], int32[:], int32[:], int64, int64",
                      locals=dict(tmp_element=float64, tmp_index=int32), nopython=True)
def _swap(heap, h2l, l2h, i, j):
    # Swap elements
    tmp_element = heap[i]
    heap[i] = heap[j]
    heap[j] = tmp_element

    # Update nodes location in the heap
    l2h[h2l[i]] = j
    l2h[h2l[j]] = i

    # Update unique node ids location
    tmp_index = h2l[i]
    h2l[i] = h2l[j]
    h2l[j] = tmp_index

@autojit("float64[:], int32[:], int32[:], int64, int64", nopython=True)
def _sift_down(heap, h2l, l2h, i, n_elements):
    while 2*i + 1 < n_elements:
        i = i*2 + 1

        if i + 1 < n_elements and heap[i + 1] > heap[i]:
            i += 1

        if heap[(i+1)/2 - 1] < heap[i]:
            _swap(heap, h2l, l2h, i, (i+1)/2 - 1)
        else:
            break

@autojit("float64[:], int32[:], int32[:], int64", nopython=True)
def _sift_up(heap, h2l, l2h, i):
    while i > 0:
        parent_index = ((i + 1)/2) - 1

        if heap[i] > heap[parent_index]:
            _swap(heap, h2l, l2h, i, parent_index)
        else:
            break

        i = parent_index

@autojit("float64, int32, float64[:], int32[:], int32[:], int64", nopython=True)
def update_heap(element, list_index, heap, h2l, l2h, n_elements):
    # i is the heap_index
    i = l2h[list_index]

    heap[i] = element
    parent_index = (i+1)/2 - 1

    # check if we should sift-up or sift-down
    if parent_index >= 0 and heap[i] > heap[parent_index]:
        _sift_up(heap, h2l, l2h, i)
    else:
        _sift_down(heap, h2l, l2h, i, n_elements)


@autojit("float64[:], int32[:], float64[:], int32[:], int32[:], int64", nopython=True)
def batchUpdate_heap(elements, indices, heap, h2l, l2h, n_elements):
    for i, list_index in enumerate(indices):
        update_heap(elements[i], list_index, heap,  h2l, l2h, n_elements)

def create_heap(elements):
    n_elements = elements.size

    heap = np.zeros(n_elements)

    h2l = np.arange(n_elements).astype("int32")
    l2h = np.arange(n_elements).astype("int32")

    indices = np.arange(n_elements).astype("int32")
    batchUpdate_heap(elements, indices, heap, h2l, l2h, n_elements)

    return heap, h2l, l2h, n_elements

def update_maxGradient(coordinate, x, args, g_func, selection_rule):
    n_depend = args["d_index"][coordinate]
    block = args["d_matrix"][coordinate, :n_depend]
    
    if selection_rule == "heapGS":
      elements = np.abs(g_func(x, args, block))

    elif selection_rule == "heapGSL":
      L = args["lipschitz"][block]
      elements = np.abs(g_func(x, args, block)) / np.sqrt(L)


    batchUpdate_heap(elements, block, args["heap"], args["h2l"], 
                     args["l2h"], args["n_params"])
    return args
# @autojit("int32[:], float64[:, :], float64[:]", nopython=True)
# def get_gradient(index_list, A, residual):
#     pass

# @autojit("int32[:], float64[:, :], float64[:], float64[:]", nopython=True)
# def get_gradientLip(index_list, A, residual, lipschitz):
#     pass

# @autojit("int32[:], float64[:, :], float64[:], float64[:]", nopython=True)
# def update_coordinate(index_list, A, residual, lipschitz):
#     pass

# @autojit("int32[:], float64[:, :], float64[:], float64[:]", nopython=True)
# def greedyUpdate(index_list, A, residual, lipschitz):
#     pass
