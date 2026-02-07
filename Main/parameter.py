import math, pandas as pd
from scipy.stats import pearsonr

# Pearson correlation b/w 2 arrays
def pearson_correlation(d1, d2):
    x, y = pd.Series(d1), pd.Series(d2)
    C_value,_ = pearsonr(x,y)
    if math.isnan(C_value) is True: C_value = 0.0
    return C_value    # pearson correlation


# Visualize the model
def visualize(mdl, filename):

    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    from keras.utils.vis_utils import plot_model

    plot_model(mdl, to_file=filename+'.pdf', show_shapes=True, show_layer_names=True, dpi=300)
