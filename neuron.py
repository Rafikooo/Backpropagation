import numpy as np
from numpy import exp


class Neuron:
    output = None
    delta = None

    def __init__(self, n_inputs, activation='sigmoid'):
        self.weights = [np.random.random() for _ in range(n_inputs + 1)]
        if activation == 'sigmoid':
            self.__activation_function = self.__sigmoid
            self.__derivative_function = self.__sigmoid_derivative

    def activate(self, inputs):
        self.output = self.__activation_function(np.dot(self.weights[1:], inputs) + self.weights[0])

        return self.output

    def derivative(self):
        return self.__derivative_function()

    def __sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def __sigmoid_derivative(self):
        return self.output * (1 - self.output)
