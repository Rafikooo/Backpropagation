from numpy.random import random
import numpy as np
from neuron import Neuron
from datetime import datetime


class Network:
    layers = []
    layers_count = 0

    def __init__(self, eta=0.005):
        self.eta = eta

    def add(self, n_output, n_inputs=None):
        if n_inputs is None:
            n_inputs = len(self.layers[-1])
        layer = []
        for _ in range(n_output):
            layer.append(Neuron(n_inputs))

        self.layers.append(layer)
        self.layers_count += 1

    def fit(self, features, labels, epochs=5, batch_size=128):
        print(f"Learning started at: {datetime.now()}")
        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")
            loss = 0
            for _ in range(len(features)):
                index = np.random.randint(0, len(features))
                feature = features[index]
                outputs = self.__forward(feature)
                for j in range(len(self.layers[-1])):
                    loss += np.power(labels[index][j] - outputs[j], 2)
                self.__backward(labels[index])
                self.__update_weights(feature)
            print(f"Loss: {loss/features.shape[0]}")

        print(f"Learning ended at: {datetime.now()}")

    def predict(self, feature):
        outputs = self.__forward(feature)

        return outputs.index(np.max(outputs))

    def evaluate(self, test_features, test_labels):
        correct = 0
        for x, y in zip(test_features, test_labels):
            classified_digit_index = self.predict(x)
            if y[classified_digit_index] == 1:
                correct += 1

        print(f"Result on test data: {correct}/{len(test_features)}")

    def __forward(self, feature):
        inputs = feature
        for layer in self.layers:
            curr_layer_outputs = []
            for neuron in layer:
                output = neuron.activate(inputs)
                curr_layer_outputs.append(output)
            inputs = curr_layer_outputs

        return inputs

    def __backward(self, label):
        for idx, neuron in enumerate(self.layers[-1]):
            neuron.delta = (neuron.output - label[idx]) * neuron.derivative()
        for layer_index in reversed(range(self.layers_count - 1)):
            layer = self.layers[layer_index]
            err = 0
            for i, neuron in enumerate(layer):
                for following_neuron in self.layers[layer_index + 1]:
                    err += following_neuron.weights[i] * following_neuron.delta * neuron.derivative()
                neuron.delta = err * neuron.derivative()

    def __update_weights(self, feature):
        for i in range(self.layers_count):
            inputs = feature if i == 0 else [neuron.output for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                for j in range(1, len(inputs)):
                    neuron.weights[j] -= self.eta * neuron.delta * inputs[j]
                neuron.weights[0] -= self.eta * neuron.delta
