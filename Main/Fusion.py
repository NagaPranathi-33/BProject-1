import sys, numpy as np

# import os

# # Add the project root directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # Now import 
  
from Main.parameter import pearson_correlation
from Main import Sort, DRN

# pearson correlation
def get_corr(data, mean):
    corr = []
    for i in range(len(data)):
        corr.append(pearson_correlation(data[i], mean))  # pearson correlation function
    return corr


# Mean calculation
def get_mean(data):
    avg = []
    for i in range(len(data)): avg.append(np.mean(data[i]))  # mean data of attributes of same class
    return avg


# Training data generation for DRN
def preprocess(data, clas):
    tr_data, tr_label = [], []
    for i in range(len(np.unique(clas))):
        pre_data = []
        for j in range(len(data)):
            if clas[j] == clas[i]:
                pre_data.append(data[j])
        mean = get_mean(np.transpose(pre_data))  # mean
        correlate = get_corr(pre_data, mean)  # pearson correlation (data, mean)
        for j in range(len(pre_data)):
            tr_data.append(pre_data[j])
            tr_label.append(correlate[j])
    return tr_data, tr_label


def update_progress(job_title, progress):
    length = 20  # modify this to change the length
    block = int(round(length * progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#" * block + "-" * (length - block), round(progress * 100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush( )


def feature(data, S, clas):

    # arrange features by pearson correlation
    print("\t>> pearson correlation based feature sorting..")
    f = Sort.by_correlate(data, clas).tolist( )

    print("\t>> Feature Fusion using Deep Residual Network..wait..")
    T = len(f[0])
    F_new, l = [], round(T)
    # training data generation for Deep Residual Network
    train_x, train_y = preprocess(data, clas)

    alpha = DRN.classify(np.array(train_x), np.array(train_y), np.array(data))  # alpha value prediction by Deep Residual Network
    for m in range(len(f)):  # data size
        update_progress("\t\tFusion", m / (len(f) - 1))
        FF = []
        for n in range(S):  # n_features to be fused
            summ, i, g = 0, n, 1
            while g <= l:  # n attributes to fused as 1
                summ += ((alpha[m][0] / g)*f[m][i])
                if (i + S) < len(f[m]): i = i + S # S = N/l
                else: g += l  # last set
                g += 1
            FF.append(summ)
        F_new.append(FF)
    return F_new
