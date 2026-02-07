import numpy as np

# import sys
# import os

# # Add the project root directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # Now import 

from Main.parameter import pearson_correlation


def by_correlate(data, clas):

    data = np.transpose(data)   # transpose for attribute selection

    # pearson correlation value
    corr_value = []
    for i in range(len(data)):
        corr_value.append(pearson_correlation(data[i], clas)) # pearson correlation value of attributes

    # Sort attribute by max
    des_corr = corr_value.copy()
    des_corr.sort()
    des_corr.reverse()     # sort the pearson correlation value in descending order

    # sorted index
    sort_index = []
    for i in range(len(des_corr)):
        ind = corr_value.index(des_corr[i])     # index of sorted data
        sort_index.append(ind)
        corr_value[ind] = -10000.0    # set by min value to avoid repeated value confusion

    # sort Attributes by sorted index
    sorted_attr = []
    for i in range(len(sort_index)):
        sorted_attr.append(data[sort_index[i]])   # add attribute according to the sorted index

    return np.transpose(sorted_attr)    # transpose to get its original form
