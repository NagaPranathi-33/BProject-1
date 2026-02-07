import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from Proposed_SSPO_DQN import SSPO

# Constants
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '2x256'
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

import logging
tf.get_logger().setLevel(logging.ERROR)


# ---------------------------------------------------------
# DQN AGENT (MODIFIED ONLY INTERNALLY)
# ---------------------------------------------------------
class DQNAgent:
    def __init__(self, train_data, train_label, test_data, test_label, pred):
        # keep pred list reference, create_model will append predictions to it
        self.pred_out = pred
        self.create_model(train_data, train_label, test_data, test_label, pred)

    def create_model(self, train_data, train_label, test_data, test_label, pred):
        # Normalize the input data
        train_data = np.array(train_data).astype(np.float32)
        test_data = np.array(test_data).astype(np.float32)
        max_train = np.max(train_data) if np.max(train_data) != 0 else 1.0
        max_test  = np.max(test_data)  if np.max(test_data)  != 0 else max_train
        train_data /= max_train
        test_data  /= max_test

        # Expand dims for Conv1D
        train_x = train_data[:, :, np.newaxis]
        test_x  = test_data[:, :, np.newaxis]

        # One-hot labels
        train_y = to_categorical(train_label)
        test_y  = to_categorical(test_label)

        # Build the CNN model
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='relu',
                         input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(train_y.shape[1], activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        # Get initial weights and total weight count
        init_weights = model.get_weights()
        total_weights = int(sum(np.prod(w.shape) for w in init_weights))

        # Call SSPO optimizer with expected weight count
        flat_opt_weights = None
        try:
            flat_opt_weights = SSPO.algm(total_weights=total_weights)
            flat_opt_weights = np.array(flat_opt_weights).flatten()
        except Exception as e:
            print(f"[SSPO] call failed: {e}")
            flat_opt_weights = None

        # Validate returned optimizer vector; fallback to current weights if mismatch
        if flat_opt_weights is None or flat_opt_weights.ndim != 1 or flat_opt_weights.size != total_weights:
            print(f"[SSPO] Warning: SSPO returned invalid weights (expected {total_weights}). Using current model init weights.")
            # flatten init_weights to vector, then back to shapes (i.e. use init_weights)
            # we will simply keep init_weights as-is (no change)
        else:
            # Reshape and set weights
            reshaped_weights = []
            pointer = 0
            for w in init_weights:
                size = int(np.prod(w.shape))
                reshaped_weights.append(flat_opt_weights[pointer:pointer+size].reshape(w.shape))
                pointer += size
            try:
                model.set_weights(reshaped_weights)
            except Exception as e:
                print(f"[SSPO] Could not set weights: {e}. Using initial weights instead.")

        # Train model (guard batch_size <= dataset size)
        safe_batch = min(1000, max(1, int(len(train_x))))
        model.fit(train_x, train_y, epochs=5, batch_size=safe_batch, verbose=0)

        # Predict and append class indices to pred list
        predictions = np.argmax(model.predict(test_x, verbose=0), axis=1)
        pred.extend(predictions.tolist())

        # return predictions for convenience (not used elsewhere)
        return predictions


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def cal_metrics(xx, yy, tpr, A, Tpr, Tnr):
    # Ensure tr is initialized correctly before any calculations that use it
    tr = tpr / 100.0
    xx = np.array(xx)
    yy = np.array(yy)

    # Ensure there are enough samples for a split
    if len(xx) < 2: # Need at least 2 samples to split into train/test
        print("[Warning] Not enough samples for a train-test split. Skipping metrics calculation.")
        A.append(0.0)
        Tpr.append(0.0)
        Tnr.append(0.0)
        return

    # Guard against invalid tr value that would result in empty train or test set
    min_samples_for_split = 1 # At least one sample for train and one for test

    train_size_calculated = int(len(xx) * tr)
    test_size_calculated = len(xx) - train_size_calculated

    if train_size_calculated < min_samples_for_split or test_size_calculated < min_samples_for_split:
        if len(xx) >= 2: # Try to force a 50/50 split if other proportions are problematic
            tr = 0.5
            print(f"[Warning] Original training proportion {tpr}% would result in an empty train or test set. "
                  f"Adjusting to 50% for train-test split.")
        else: # Still not enough samples even for 50/50
            print("[Warning] Not enough samples for a valid train-test split even after adjustment. Skipping metrics calculation.")
            A.append(0.0)
            Tpr.append(0.0)
            Tnr.append(0.0)
            return


    # Safe train_test_split now without stratify
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)

    pred = []
    DQNAgent(x_train, y_train, x_test, y_test, pred)
    predict = np.array(pred)
    target = np.array(y_test)

    # If prediction length mismatches test set, fix by trimming/padding (defensive)
    if predict.shape[0] != target.shape[0]:
        print(f"[Warning] predictions ({predict.shape[0]}) != y_test ({target.shape[0]}). Trimming to min length.")
        m = min(predict.shape[0], target.shape[0])
        predict = predict[:m]
        target  = target[:m]

    # Accuracy
    acc = float(accuracy_score(target, predict))
    A.append(acc)

    # TPR = average recall (sensitivity) across classes
    try:
        tpr_val = float(recall_score(target, predict, average='macro', zero_division=0))
    except Exception:
        tpr_val = 0.0
    Tpr.append(tpr_val)

    # TNR (specificity) average across classes
    classes = np.unique(np.concatenate([target, predict]))
    cm = confusion_matrix(target, predict, labels=classes)
    total = cm.sum()
    spec_list = []
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FP - FN
        denom = (TN + FP)
        spec = (TN / denom) if denom > 0 else 0.0
        spec_list.append(spec)
    tnr_val = float(np.mean(spec_list)) if len(spec_list) > 0 else 0.0
    Tnr.append(tnr_val)
