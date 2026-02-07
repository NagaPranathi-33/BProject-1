import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# import sys
# import os
# # Add the project root directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # Now import 
from Adaptive_EBat_DBN.MLP import MLP
from Adaptive_EBat_DBN.RBM import RBM
from Adaptive_EBat_DBN import bat

from sklearn.model_selection import train_test_split
class DBN(object):
	def __init__(self, layers, n_labels):
		self.rbms = []
		self.n_labels = n_labels
		for n_v, n_h in zip(layers[:-1], layers[1:]):
			self.rbms.append(RBM(n_v, n_h, epochs=2, lr=0.1))
		self.mlp = MLP(act_type='Sigmoid', opt_type='Adam', layers=layers+[n_labels], epochs=5, learning_rate=0.01, lmbda=1e-2)

	def pretrain(self, x, ow):
		v = x
		for rbm in self.rbms:
			rbm.fit(v, ow)
			v = rbm.marginal_h(v)

	def finetuning(self, x, labels):
		# assign weights
		self.mlp.w = [rbm.w for rbm in self.rbms] + [np.random.randn(self.rbms[-1].w.shape[1], self.n_labels)]
		self.mlp.b = [rbm.b for rbm in self.rbms] + [np.random.randn(1, self.n_labels)]
		self.mlp.fit(x, labels)

	def fit(self, x, y, opt_w):
		self.pretrain(x, opt_w)
		self.finetuning(x, y)

	def predict(self, x):
		return self.mlp.predict(x)

'''def classify(xx,yy,tr,A,Tpr,Tnr):
	# split data by class label
	def train_test_split(data, clas, tr_per):
		train_x, train_y = [], []  # training data, training class
		test_x, test_y, label = [], [], []  # testing data, testing class, label
		uni = np.unique(clas)  # unique class
		for i in range(len(uni)):  # n_unique class
			tem = []
			for j in range(len(clas)):
				if (uni[i] == clas[j]):  # if class of data = unique class
					tem.append(data[j])  # get unique class as tem
			tp = int((len(tem) * tr_per) / 100)  # training data size
			for k in range(len(tem)):
				if (k < tp):  # adding training data & its class
					train_x.append(tem[k])
					train_y.append(uni[i])
					label.append(uni[i])
				else:  # adding testing data & its class
					test_x.append(tem[k])
					test_y.append(uni[i])
					label.append(uni[i])
		return train_x, train_y, test_x, test_y, label

	train_x, train_y, test_x, test_y, target = train_test_split(xx, yy, tr)
	trp = tr/100
	opt_w = bat.algm()
	tp, tn, fn, fp = 0, 0, 0, 0
	trainy = []
	for i in range(len(train_y)):
		trainy.append((int)(train_y[i]))

	dbn = DBN([np.array(train_x).shape[1], 10, 10], (len(np.unique(test_y))))	# layers, n_labels

	dbn.fit(np.array(train_x), np.array(trainy), opt_w)
	pred = np.argmax(dbn.predict(np.array(test_x)), axis=1)
	target = test_y
	predict = []
	for i in range(len(pred)):
		predict.append(pred[i])

	unique_clas = np.unique(test_y)
	for i1 in range(len(unique_clas)):
		c = unique_clas[i1];
		for i in range(len(target)):
			if (target[i] == c and predict[i] == c):
				tp += 1
			if (target[i] != c and predict[i] != c):
				tn += 1
			if (target[i] == c and predict[i] != c):
				fn += 1
			if (target[i] != c and predict[i] == c):
				fp += 1
	tn = tn / len(unique_clas)
	fn = fn / len(unique_clas)
	fp = fp / len(unique_clas)
	A.append((tp + tn) / (tp + tn + fp + fn))
	Tpr.append(tp / (tp + fn))
	Tnr.append(tn / (tn + fp))'''

def classify(xx, yy, tr, A, Tpr, Tnr):
    # Split data by class label
    def train_test_split(data, clas, tr_per):
        train_x, train_y = [], []  # training data, training class
        test_x, test_y, label = [], [], []  # testing data, testing class, label
        uni = np.unique(clas)  # unique class

        for i in range(len(uni)):  # n_unique class
            tem = []
            for j in range(len(clas)):
                if uni[i] == clas[j]:  # if class of data = unique class
                    tem.append(data[j])  # get unique class as tem
            tp = int((len(tem) * tr_per) / 100)  # training data size
            for k in range(len(tem)):
                if k < tp:  # adding training data & its class
                    train_x.append(tem[k])
                    train_y.append(uni[i])
                    label.append(uni[i])
                else:  # adding testing data & its class
                    test_x.append(tem[k])
                    test_y.append(uni[i])
                    label.append(uni[i])

        return train_x, train_y, test_x, test_y, label

    train_x, train_y, test_x, test_y, target = train_test_split(xx, yy, tr)

    # ✅ Debugging the data shapes
    print(f"train_x shape: {np.array(train_x).shape}")
    print(f"train_y shape: {np.array(train_y).shape}")
    print(f"test_x shape: {np.array(test_x).shape}")
    print(f"test_y shape: {np.array(test_y).shape}")

    if len(train_x) == 0 or len(test_x) == 0:
        print("❌ Error: Train or test data is empty. Please check the dataset.")
        return

    trp = tr / 100
    opt_w = bat.algm()

    # ✅ Check if opt_w is valid
    if opt_w is None or not isinstance(opt_w, np.ndarray):
        print("❌ Error: opt_w is invalid or None.")
        return

    tp, tn, fn, fp = 0, 0, 0, 0
    trainy = [int(y) for y in train_y]

    # ✅ Ensure train_x has the correct shape for DBN
    input_size = np.array(train_x).shape[1] if len(train_x) > 0 else 10  # Fallback to 10 if empty
    dbn = DBN([input_size, 10, 10], len(np.unique(test_y)))  # layers, n_labels

    # ✅ Fit DBN model (check for errors)
    try:
        dbn.fit(np.array(train_x), np.array(trainy), opt_w)
    except Exception as e:
        print(f"❌ Exception in DBN fit: {e}")
        return

    pred = np.argmax(dbn.predict(np.array(test_x)), axis=1)
    target = test_y
    predict = list(pred)

    unique_clas = np.unique(test_y)
    for i1 in range(len(unique_clas)):
        c = unique_clas[i1]
        for i in range(len(target)):
            if target[i] == c and predict[i] == c:
                tp += 1
            if target[i] != c and predict[i] != c:
                tn += 1
            if target[i] == c and predict[i] != c:
                fn += 1
            if target[i] != c and predict[i] == c:
                fp += 1

    tn = tn / len(unique_clas) if len(unique_clas) > 0 else 1
    fn = fn / len(unique_clas) if len(unique_clas) > 0 else 1
    fp = fp / len(unique_clas) if len(unique_clas) > 0 else 1

    A.append((tp + tn) / (tp + tn + fp + fn))
    Tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    Tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

