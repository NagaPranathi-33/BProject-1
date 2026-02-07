import random
import numpy as np
import itertools

def data_aug(input_data, clas):  # oversampling
    total_instance = 25000  # total number of required rows
    input_data = list(itertools.chain(*input_data))
    clas = list(itertools.chain(*clas))

    def augment(data, lab, ins_total):
        c_min, c_max = [], []

        for i in range(len(data[0])):  # For each feature
            col = np.array(data)[:, i]
            c_min.append(np.min(col))
            c_max.append(np.max(col))

        augmented_data = []
        augmented_labels = []

        for i in range(ins_total):
            if i < len(data):
                augmented_data.append(data[i])
                augmented_labels.append(lab[i])
            else:
                temp_instance = [random.uniform(c_min[j], c_max[j]) for j in range(len(data[0]))]
                random_label = random.choice(lab)  # pick label randomly from existing
                augmented_data.append(temp_instance)
                augmented_labels.append(random_label)

        return augmented_data, augmented_labels

    Data, clas = augment(input_data, clas, total_instance)
    return np.array(Data), np.array(clas)
