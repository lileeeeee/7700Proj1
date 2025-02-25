import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    n_samples = train_x.shape[0]
    for i in range(0, n_samples, batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]
        yield batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) * (1 - self.forward(x))


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) * (1 - self.forward(x))


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.log(1 + np.exp(x))) + x * (1 - np.tanh(np.log(1 + np.exp(x))) ** 2) * (1 / (1 + np.exp(-x)))

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_pred + 1e-8))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))

        # this will store the activations (forward prop)
        self.z = None
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None
        self.dropout_mask = None


    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        self.z = np.dot(h, self.W) + self.b
        a = self.activation_function.forward(self.z)
        if training and self.dropout_rate > 0.0:
            # Create dropout mask: 1 where we keep the neuron, 0 where we drop it
            self.dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32)
            a *= self.dropout_mask
        else:
            self.dropout_mask = np.ones_like(a)
        self.activations = a
        # print(f"Layer i: z range=({self.z.min()}, {self.z.max()})")
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        local_grad = self.activation_function.derivative(self.z) * self.dropout_mask
        local_delta = delta * local_grad

        # Compute gradients for weights and biases
        dL_dW = np.dot(h.T, local_delta)
        dL_db = np.sum(local_delta, axis=0, keepdims=True)
        # Compute delta to pass to the previous layer
        prev_delta = np.dot(local_delta, self.W.T)
        self.delta = local_delta
        return dL_dW, dL_db, prev_delta


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        h = x
        for layer in self.layers:
            h = layer.forward(h, training)
        return h

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        # Store the activations from the forward pass (including the network input)
        activations = [input_data]
        for layer in self.layers:
            activations.append(layer.activations)

        dW_all = []
        db_all = []
        delta = loss_grad
        # Propagate backwards through the layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h_prev = activations[i]  # Activation from the previous layer
            dW, db, delta = layer.backward(h_prev, delta)
            # Insert gradients at the beginning so that order matches the forward pass
            dW_all.insert(0, dW)
            db_all.insert(0, db)

        return dW_all, db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, rmsprop: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the network using mini-batch stochastic gradient descent
        """
        training_losses = []
        validation_losses = []

        # RMSProp parameters and caches
        if rmsprop:
            decay_rate = 0.9
            eps = 1e-8
            cache_W = [np.zeros_like(layer.W) for layer in self.layers]
            cache_b = [np.zeros_like(layer.b) for layer in self.layers]

        for epoch in range(epochs):
            batch_losses = []
            # Iterate over mini-batches using a generator
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                predictions = self.forward(batch_x, training=True)

                loss = loss_func.loss(batch_y, predictions)

                # print("loss", loss)
                # exit(0)
                batch_loss = np.mean(loss)
                batch_losses.append(batch_loss)

                # Compute the gradient of the loss function
                loss_grad = loss_func.derivative(batch_y, predictions)
                dW_all, db_all = self.backward(loss_grad, batch_x)


                # Update weights and biases for each layer
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        # Update RMSProp caches
                        cache_W[i] = decay_rate * cache_W[i] + (1 - decay_rate) * (dW_all[i] ** 2)
                        cache_b[i] = decay_rate * cache_b[i] + (1 - decay_rate) * (db_all[i] ** 2)
                        # Update parameters using RMSProp update rule
                        layer.W -= learning_rate * dW_all[i] / (np.sqrt(cache_W[i]) + eps)
                        layer.b -= learning_rate * db_all[i] / (np.sqrt(cache_b[i]) + eps)
                    else:
                        # Vanilla SGD update
                        layer.W -= learning_rate * dW_all[i]
                        layer.b -= learning_rate * db_all[i]

            # Average training loss
            epoch_train_loss = np.mean(batch_losses)
            training_losses.append(epoch_train_loss)
            val_predictions = self.forward(val_x, training=False)
            val_loss = np.mean(loss_func.loss(val_y, val_predictions))
            validation_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        return np.array(training_losses), np.array(validation_losses)
