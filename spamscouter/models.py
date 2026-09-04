import numpy as np
from tqdm import tqdm, trange
from math import ceil
import pickle
from abc import ABC
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data


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
    MAX_EPOCHS = 1000
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 3.0
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    PROBABILITY_EPSILON = 1e-7
    VARIANCE_EPSILON = 1e-12
    PARAMETERS = ('W1_', 'b1_', 'W2_', 'b2_')
    DECAYED_PARAMETERS = ('W1_', 'W2_')

    @staticmethod
    def _sigmoid(x):
        exponential = np.exp(-np.abs(x))
        return np.where(x >= 0, 1 / (1 + exponential), exponential / (1 + exponential))

    @classmethod
    def _sigmoid_prime(cls, x):
        x = cls._sigmoid(x)
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

    def __init__(self, hidden_layer_size=100, final_activation_function='sigmoid',
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, balance_classes=True,
                 random_state=None):
        self.hidden_layer_size = hidden_layer_size
        self.final_activation_function = final_activation_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.balance_classes = balance_classes
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # a probability model rather than a general regressor: the output is bounded to
        # [0, 1], so it cannot score well against sklearn's arbitrary continuous targets
        tags.regressor_tags.poor_score = True
        return tags

    def _initialise(self, input_size):
        if self.final_activation_function not in self.FINAL_ACTIVATION_FUNCTIONS:
            raise ValueError(f'unknown final activation function: {self.final_activation_function}')

        if self.hidden_layer_size < 1:
            raise ValueError(f'hidden_layer_size must be at least 1, got {self.hidden_layer_size}')

        # decoupled decay shrinks the weights by (1 - learning_rate * weight_decay) every
        # step, so at 1 they collapse to zero and beyond it they alternate sign and diverge
        if self.learning_rate * self.weight_decay >= 1:
            raise ValueError(
                'learning_rate * weight_decay must be below 1, got '
                f'{self.learning_rate} * {self.weight_decay} = {self.learning_rate * self.weight_decay}'
            )

        self.rng_ = np.random.default_rng(self.random_state)

        self.W1_ = self.rng_.normal(size=(self.hidden_layer_size, input_size)) * np.sqrt(2 / input_size)
        self.b1_ = np.zeros((self.hidden_layer_size, 1))
        self.a1_ = self._relu
        self.a1p_ = self._relu_prime

        if self.final_activation_function == 'sigmoid':
            self.a2_ = self._sigmoid
            self.a2p_ = self._sigmoid_prime
            self.W2_ = self.rng_.normal(size=self.hidden_layer_size) * np.sqrt(1 / self.hidden_layer_size)

        if self.final_activation_function == 'linear':
            self.a2_ = self._linear
            self.a2p_ = self._linear_prime
            self.W2_ = self.rng_.normal(size=self.hidden_layer_size) * np.sqrt(1 / self.hidden_layer_size)

        if self.final_activation_function == 'clipped':
            self.a2_ = self._clipped
            self.a2p_ = self._clipped_prime
            self.W2_ = self.rng_.normal(size=self.hidden_layer_size) * np.sqrt(1 / self.hidden_layer_size)

        self.b2_ = np.zeros(1)

        # identity transform and neutral weights until fit() measures the training split
        self.input_mean_ = np.zeros((input_size, 1))
        self.input_std_ = np.ones((input_size, 1))
        self.positive_weight_ = 1.0
        self.negative_weight_ = 1.0

    def predict(self, X):
        check_is_fitted(self)
        vectors = validate_data(self, X, reset=False, dtype=np.float64)
        return self._forward(self._standardise(vectors.T))[3]

    def _standardise(self, vectors):
        return (vectors - self.input_mean_) / self.input_std_

    def _split_stratified(self, labels):
        validation_indices = []
        training_indices = []

        for class_indices in (np.flatnonzero(labels <= 0.5), np.flatnonzero(labels > 0.5)):
            class_indices = self.rng_.permutation(class_indices)

            if class_indices.size < 2:
                training_indices.append(class_indices)
                continue

            split_idx = int(round(class_indices.size * self.VALIDATION_FRACTION))
            split_idx = min(max(split_idx, 1), class_indices.size - 1)
            validation_indices.append(class_indices[:split_idx])
            training_indices.append(class_indices[split_idx:])

        validation_indices = np.concatenate(validation_indices) if validation_indices else np.empty(0, dtype=int)
        training_indices = np.concatenate(training_indices)

        if validation_indices.size == 0:
            # every class had a single member; fall back to holding one sample out
            training_indices = self.rng_.permutation(training_indices)
            validation_indices, training_indices = training_indices[:1], training_indices[1:]

        return validation_indices, training_indices

    def _forward(self, vectors):
        z1 = self.W1_ @ vectors + self.b1_
        a1 = self.a1_(z1)
        z2 = self.W2_ @ a1 + self.b2_
        a2 = self.a2_(z2)
        return z1, a1, z2, a2

    @classmethod
    def _binary_cross_entropy(cls, predictions, labels, weights=1):
        predictions = np.clip(predictions, cls.PROBABILITY_EPSILON, 1 - cls.PROBABILITY_EPSILON)
        return -np.mean(weights * (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)))

    def _sample_weights(self, labels):
        return np.where(labels > 0.5, self.positive_weight_, self.negative_weight_)

    def _gradients(self, vectors, labels, weights=1):
        z1, a1, z2, a2 = self._forward(vectors)

        # d(BCE)/d(a2). For the sigmoid output the a2 * (1 - a2) denominator here is
        # exactly what a2p returns below, so the two cancel and this reduces to the
        # usual (a2 - labels); keeping it explicit is what lets every final activation
        # share one backward pass.
        predictions = np.clip(a2, self.PROBABILITY_EPSILON, 1 - self.PROBABILITY_EPSILON)
        delta2 = weights * (predictions - labels) / (predictions * (1 - predictions)) / labels.size
        delta2 = delta2 * self.a2p_(z2)
        delta1 = np.outer(self.W2_, delta2) * self.a1p_(z1)

        return {
            'W1_': delta1 @ vectors.T,
            'b1_': delta1.sum(axis=1, keepdims=True),
            'W2_': a1 @ delta2,
            'b2_': np.atleast_1d(delta2.sum()),
        }

    def fit(self, X, y):
        # sklearn requires these exact parameter names; everything downstream of the
        # validation call keeps the descriptive ones
        vectors, labels = validate_data(self, X, y, dtype=np.float64, y_numeric=True)

        if vectors.shape[0] < 2:
            raise ValueError(f'need at least two samples to fit, got n_samples={vectors.shape[0]}')

        self._initialise(vectors.shape[1])
        vectors = vectors.T
        labels = labels.astype(float)

        validation_indices, training_indices = self._split_stratified(labels)

        valid_vectors = vectors[:, validation_indices]
        train_vectors = vectors[:, training_indices]
        valid_labels = labels[validation_indices]
        train_labels = labels[training_indices]

        # statistics come from the training split only, so the validation split stays honest
        self.input_mean_ = train_vectors.mean(axis=1, keepdims=True)
        self.input_std_ = train_vectors.std(axis=1, keepdims=True)
        self.input_std_[self.input_std_ < self.VARIANCE_EPSILON] = 1

        valid_vectors = self._standardise(valid_vectors)
        train_vectors = self._standardise(train_vectors)

        # The HPO cache is deliberately 50/50 but real mailboxes are not, and
        # spamassassin.cf keys off absolute probability thresholds. Re-weighting each
        # class to half the total keeps the output centred the same way regardless of
        # the incoming ratio -- and evaluates to exactly 1.0 on balanced data, so it
        # cannot perturb an HPO run.
        if self.balance_classes:
            positives = np.count_nonzero(train_labels > 0.5)
            negatives = train_labels.size - positives

            # with only one class present there is nothing to rebalance against, and
            # weighting it would just rescale the loss and skew early stopping
            if positives and negatives:
                self.positive_weight_ = train_labels.size / (2 * positives)
                self.negative_weight_ = train_labels.size / (2 * negatives)

        train_weights = self._sample_weights(train_labels)
        valid_weights = self._sample_weights(valid_labels)

        beta1, beta2 = self.ADAM_BETAS
        moment1 = {name: np.zeros_like(getattr(self, name)) for name in self.PARAMETERS}
        moment2 = {name: np.zeros_like(getattr(self, name)) for name in self.PARAMETERS}
        timestep = 0

        incumbent_loss = np.inf
        incumbent_parameters = {name: getattr(self, name).copy() for name in self.PARAMETERS}
        epochs_without_new_incumbent = 0

        with tqdm(desc='classifier Epochs', total=self.MAX_EPOCHS) as progress:
            for _ in range(self.MAX_EPOCHS):
                epoch_perm = self.rng_.permutation(train_vectors.shape[1])
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

                    gradients = self._gradients(train_vectors[:, batch_indices], train_labels[batch_indices], train_weights[batch_indices])
                    timestep += 1

                    for name, gradient in gradients.items():
                        moment1[name] = beta1 * moment1[name] + (1 - beta1) * gradient
                        moment2[name] = beta2 * moment2[name] + (1 - beta2) * gradient ** 2
                        corrected1 = moment1[name] / (1 - beta1 ** timestep)
                        corrected2 = moment2[name] / (1 - beta2 ** timestep)

                        value = getattr(self, name)
                        update = corrected1 / (np.sqrt(corrected2) + self.ADAM_EPSILON)
                        if name in self.DECAYED_PARAMETERS:
                            update = update + self.weight_decay * value

                        setattr(self, name, value - self.learning_rate * update)

                progress.update(1)
                eval_loss = self._binary_cross_entropy(self._forward(valid_vectors)[3], valid_labels, valid_weights)

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
