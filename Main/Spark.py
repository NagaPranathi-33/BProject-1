from pyspark.conf import SparkConf
import os
import numpy as np
import Proposed_SSPO_DQN.DQN
from Main import DFCM, Fusion, Data_Augmentation

conf = SparkConf()
conf.setMaster("local")

def Master_Slave(data, dts, lab, tr, A, Tpr, Tnr):
    n_c = 4
    Cluster, Lab = DFCM.clustering(data, n_c, lab)  # Assuming clustering is working fine
    n_slv = 4
    feat_s = 10
    Feat = []

    # Generate features using the fusion method
    for i in range(n_slv):
        print("Slave::", (i + 1))
        Feat.append(Fusion.feature(Cluster[i], feat_s, Lab[i]))

    print("Data Augmentation...")
    Data, clas = [], []

    # Apply data augmentation to each cluster's features
    for i in range(n_slv):
        aug_data, aug_label = Data_Augmentation.data_aug([Feat[i]], [Lab[i]])
        Data.append(aug_data)
        clas.append(aug_label)

    # Flatten the Data and labels before passing to cal_metrics
    Data = np.concatenate(Data, axis=0)  # Flatten all augmented data
    clas = np.concatenate(clas, axis=0)  # Flatten all augmented labels

    # Ensure the data and labels have the correct shape
    print(f"Data shape after augmentation: {Data.shape}")
    print(f"Labels shape after augmentation: {clas.shape}")

    # Save preprocessed data for future use
    preprocessed_dir = "Preprocessed"
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    np.save(os.path.join(preprocessed_dir, f"{dts}_data.npy"), Data)
    np.save(os.path.join(preprocessed_dir, f"{dts}_class.npy"), clas)

    # Call the cal_metrics function to calculate the metrics
    Proposed_SSPO_DQN.DQN.cal_metrics(Data, clas, tr, A, Tpr, Tnr)

    return A, Tpr, Tnr, Data, clas
