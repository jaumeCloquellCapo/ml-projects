import numpy as np


# our sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y - y ** 2


class MLP():
    """
    A class used to represent out Neural Networks with 2 Layers (Input and hidden)

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

        for i in range(self.ni-1):
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