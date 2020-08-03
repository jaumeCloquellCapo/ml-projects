import numpy as np


# our sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y - y ** 2

def cost_derivative (output_activations , y):
    return output_activations - y


class MLP:
    """
    A class used to represent multiple single neurons to a multi-layer feedforward neural network; this special type of network is also called a multi-layer perceptron (MLP).

    ...

    Attributes
    ----------
    ni : int
        input nodes
    nh : int
        hidden nodes
    no : int
        output nodes

    Methods
    -------
    activation(inputs)
        Mult. the input values with the heights and apply sigmoid function
    """

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # weights
        self.wi = np.random.random((self.ni, self.nj))
        self.wo = np.random.random((self.nj, self.no))

        # last change in weights for momentum
        self.ci = np.zeros(self.ni, self.nh)
        self.co = np.zeros(self.nh, self.no)

    def activation(self, inputs):

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum_h = 0.0
            for i in range(self.ni):
                sum_h += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum_h)

        # output activations
        for k in range(self.no):
            sum_o = 0.0
            for j in range(self.nh):
                sum_o += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum_o)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        """ Update the network ’s weights and biases by applying gradient descent using
        backpropagation"""
        output_deltas = np.zeros(self.no)
        for i in range(self.no):
            error = cost_derivative(targets[i], self.co[i])
            output_deltas[i] = dsigmoid(self.ao[i]) * error

        hidden_deltas = np.zeros(self.nh)
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[i] = dsigmoid(self.nh[i]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
            self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
            self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error
