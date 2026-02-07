# import all necessery libraries
from numpy import exp, array, random, dot, tanh
from sklearn.model_selection import train_test_split
import numpy as np

# Class to create a neural
# network with single neuron
class NeuralNetwork():

    def __init__(self, dim):
        # Using seed to make sure it'll
        # generate same weights in every run
        random.seed(1)

        # 3x1 Weight matrix
        self.weight_matrix = 2 * random.random((dim, 1)) - 1
        self.weight_matrix = self.weight_matrix # weight update
    # tanh as activation function
    def tanh(self, x):
        return tanh(x)

        # derivative of tanh function.

    # Needed to calculate the gradients.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2

    # forward propagation
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

        # training the neural network.

    def train(self, train_inputs, train_outputs,
              num_train_iterations):
        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)

            # Calculate the error in the output.
            error = train_outputs - output

            # multiply the error by input and then
            # by gradient of tanh funtion to calculate
            # the adjustment needs to be made in weights
            adjustment = dot(train_inputs.T, error *
                             self.tanh_derivative(output))

            # Adjust the weight matrix
            self.weight_matrix += adjustment

        # Driver Code


def classify(x1, y1,tr, A,Tpr,Tnr):
    tr=tr/100
    x = np.asarray(x1)
    y = np.asarray(y1)
    train_inputs, test_inputs, y_train, y_test = train_test_split(x, y, train_size=tr)
    train_outputs = np.array([y_train.tolist()]).T

    neural_network = NeuralNetwork(len(train_inputs[0]))
    neural_network.train(train_inputs, train_outputs, 10000)

    # Test the neural network with a new situation.
    pred = neural_network.forward_propagation(test_inputs)
    predict=[]
    for i in range(len(pred)):
        if round(np.abs(pred[i][0])) < 0:
            predict.append(0)
        else:
            predict.append(1)
    target = np.concatenate((y_test,y_train))
    Scores =  np.concatenate((predict,y_train))
    unique_clas = np.unique(target)
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(unique_clas)):
        c = unique_clas[i1]
        for i in range(len(Scores)):
            if (target[i] == c and Scores[i] == c):
                tp = tp + 1
            if (target[i] != c and Scores[i] != c):
                tn = tn + 1
            if (target[i] == c and Scores[i] != c):
                fn = fn + 1
            if (target[i] != c and Scores[i] == c):
                fp = fp + 1

    tn = tn / len(unique_clas)

    acc = (tp + tn) / (tp + tn + fp + fn)
    A.append(acc)
    A.sort()
    Tpr.append(tp / (tp + fn))
    Tpr.sort()
    Tnr.append(tn / (tn + fp))
    Tnr.sort()
