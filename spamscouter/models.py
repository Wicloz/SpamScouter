import numpy as np
from tqdm import tqdm, trange
from math import ceil
import pickle
from abc import ABC
from sklearn.base import RegressorMixin, BaseEstimator


class SpamRegressorMixin(ABC):
    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)

    def predict(self, vectors):
        return np.clip(super().predict(vectors), 0, 1)

    def fit(self, seed, vectors, labels):
        super().fit(vectors, labels)

    def accuracy(self, vectors, labels):
        predictions = self.predict(vectors)
        predictions[labels == False] = 1 - predictions[labels == False]
        return np.mean(predictions)

    @staticmethod
    def hyper_parameter_space():
        return []


class NeuralNetworkRegressor(RegressorMixin, BaseEstimator):
    FINAL_ACTIVATION_FUNCTIONS = ('sigmoid', 'linear', 'clipped')

    BATCH_SIZE = 100
    VALIDATION_FRACTION = 0.1
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.001
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    PROBABILITY_EPSILON = 1e-7
    PARAMETERS = ('W1', 'b1', 'W2', 'b2')

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_prime(x):
        x = 1 / (1 + np.exp(-x))
        return x * (1 - x)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_prime(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def _linear(x):
        return x

    @staticmethod
    def _linear_prime(x):
        return np.ones(x.shape, x.dtype)

    @staticmethod
    def _clipped(x):
        return np.clip(x, 0, 1)

    @staticmethod
    def _clipped_prime(x):
        return ((x > 0) & (x < 1)).astype(x.dtype)

    def __init__(self, input_size, hidden_layer_size=100, final_activation_function='sigmoid', random_state=None):
        self.rng = np.random.default_rng(random_state)

        self.W1 = self.rng.normal(size=(hidden_layer_size, input_size)) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((hidden_layer_size, 1))
        self.a1 = self._relu
        self.a1p = self._relu_prime

        if final_activation_function not in self.FINAL_ACTIVATION_FUNCTIONS:
            raise ValueError(f'unknown final activation function: {final_activation_function}')

        if final_activation_function == 'sigmoid':
            self.a2 = self._sigmoid
            self.a2p = self._sigmoid_prime
            self.W2 = self.rng.normal(size=hidden_layer_size) * np.sqrt(1 / hidden_layer_size)

        if final_activation_function == 'linear':
            self.a2 = self._linear
            self.a2p = self._linear_prime
            self.W2 = self.rng.normal(size=hidden_layer_size) * np.sqrt(1 / hidden_layer_size)

        if final_activation_function == 'clipped':
            self.a2 = self._clipped
            self.a2p = self._clipped_prime
            self.W2 = self.rng.normal(size=hidden_layer_size) * np.sqrt(1 / hidden_layer_size)

        self.b2 = np.zeros(1)

    def predict(self, vectors):
        vectors = np.atleast_2d(vectors).T
        vectors = self.W1 @ vectors + self.b1
        vectors = self.a1(vectors)
        vectors = self.W2 @ vectors + self.b2
        vectors = self.a2(vectors)
        return vectors

    def _forward(self, vectors):
        z1 = self.W1 @ vectors + self.b1
        a1 = self.a1(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.a2(z2)
        return z1, a1, z2, a2

    @classmethod
    def _binary_cross_entropy(cls, predictions, labels):
        predictions = np.clip(predictions, cls.PROBABILITY_EPSILON, 1 - cls.PROBABILITY_EPSILON)
        return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    def _gradients(self, vectors, labels):
        z1, a1, z2, a2 = self._forward(vectors)

        # d(BCE)/d(a2). For the sigmoid output the a2 * (1 - a2) denominator here is
        # exactly what a2p returns below, so the two cancel and this reduces to the
        # usual (a2 - labels); keeping it explicit is what lets the other three final
        # activations share one backward pass.
        predictions = np.clip(a2, self.PROBABILITY_EPSILON, 1 - self.PROBABILITY_EPSILON)
        delta2 = (predictions - labels) / (predictions * (1 - predictions)) / labels.size
        delta2 = delta2 * self.a2p(z2)
        delta1 = np.outer(self.W2, delta2) * self.a1p(z1)

        return {
            'W1': delta1 @ vectors.T,
            'b1': delta1.sum(axis=1, keepdims=True),
            'W2': a1 @ delta2,
            'b2': np.atleast_1d(delta2.sum()),
        }

    def fit(self, vectors, labels):
        vectors = np.atleast_2d(vectors).T
        labels = np.asarray(labels, dtype=float)

        if vectors.shape[1] < 2:
            raise ValueError('need at least two samples to fit')

        train_eval_perm = self.rng.permutation(vectors.shape[1])
        vectors = vectors[:, train_eval_perm]
        labels = labels[train_eval_perm]

        train_eval_split_idx = int(round(vectors.shape[1] * self.VALIDATION_FRACTION))
        train_eval_split_idx = min(max(train_eval_split_idx, 1), vectors.shape[1] - 1)

        valid_vectors = vectors[:, :train_eval_split_idx]
        train_vectors = vectors[:, train_eval_split_idx:]
        valid_labels = labels[:train_eval_split_idx]
        train_labels = labels[train_eval_split_idx:]

        beta1, beta2 = self.ADAM_BETAS
        moment1 = {name: np.zeros_like(getattr(self, name)) for name in self.PARAMETERS}
        moment2 = {name: np.zeros_like(getattr(self, name)) for name in self.PARAMETERS}
        timestep = 0

        incumbent_loss = np.inf
        incumbent_parameters = {name: getattr(self, name).copy() for name in self.PARAMETERS}
        epochs_without_new_incumbent = 0

        with tqdm(desc='classifier Epochs') as progress:
            while True:
                epoch_perm = self.rng.permutation(train_vectors.shape[1])
                steps = ceil(train_vectors.shape[1] / self.BATCH_SIZE)
                deficit = -train_vectors.shape[1] % self.BATCH_SIZE
                deficit_per_step = ceil(deficit / steps)
                idx = 0

                for _ in trange(steps, leave=False):
                    deficit_consumed = min(deficit, deficit_per_step)
                    batch_size = self.BATCH_SIZE - deficit_consumed
                    deficit -= deficit_consumed
                    batch_indices = epoch_perm[idx:idx + batch_size]
                    idx += batch_size

                    gradients = self._gradients(train_vectors[:, batch_indices], train_labels[batch_indices])
                    timestep += 1

                    for name, gradient in gradients.items():
                        moment1[name] = beta1 * moment1[name] + (1 - beta1) * gradient
                        moment2[name] = beta2 * moment2[name] + (1 - beta2) * gradient ** 2
                        corrected1 = moment1[name] / (1 - beta1 ** timestep)
                        corrected2 = moment2[name] / (1 - beta2 ** timestep)
                        setattr(self, name, getattr(self, name) - self.LEARNING_RATE * corrected1 / (np.sqrt(corrected2) + self.ADAM_EPSILON))

                progress.update(1)
                eval_loss = self._binary_cross_entropy(self._forward(valid_vectors)[3], valid_labels)

                if eval_loss < incumbent_loss:
                    incumbent_loss = eval_loss
                    incumbent_parameters = {name: getattr(self, name).copy() for name in self.PARAMETERS}
                    epochs_without_new_incumbent = 0

                else:
                    epochs_without_new_incumbent += 1
                    if epochs_without_new_incumbent >= self.EARLY_STOPPING_PATIENCE:
                        break

        for name, value in incumbent_parameters.items():
            setattr(self, name, value)

        return self
