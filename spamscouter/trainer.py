from .connectors import ConnectorIMAP, ConnectorCache
from tokenizers import BertWordPieceTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
from tempfile import TemporaryDirectory
from shutil import move, rmtree
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, Beta
from random import Random
from .models import SpamSVM, SpamNearestNeighbors, SpamNeuralNetwork
from .message import MESSAGE_PROCESS_METHODS, infer_vector
import json
from tqdm import trange, tqdm
import numpy as np
import pickle


CONNECTORS = {
    'IMAP': ConnectorIMAP,
    'CACHE': ConnectorCache,
}


CS = ConfigurationSpace()
CS.add(Integer('document_vector_size', (100, 1000), default=938))
CS.add(Categorical('message_processing_method', MESSAGE_PROCESS_METHODS.keys(), default='unicode'))
CS.add(Float('vocab_size_per_message', (0, 2), default=1.5721482111031, distribution=Beta(4, 4)))
CS.add(Integer('vocab_token_min_count', (1, 1000), default=1, log=True))
CS.add(Integer('max_message_characters', (1000, 1000000), default=152486, log=True))
CS.add(Categorical('include_visible_headers', (True, False), default=True))

REGRESSORS = {
    'SVM': SpamSVM,
    'NearestNeighbors': SpamNearestNeighbors,
    'NeuralNetwork': SpamNeuralNetwork,
}

regressor_hp = Categorical('regressor_type', REGRESSORS.keys(), default='NeuralNetwork')
CS.add(regressor_hp)

for regressor_key, regressor_class in REGRESSORS.items():
    CS.add_configuration_space(regressor_key, regressor_class.configuration_space(), delimiter='.',
                               parent_hyperparameter={'parent': regressor_hp, 'value': regressor_key})


class Trainer:
    def __init__(self, settings):
        self.settings = settings

    def _make_tokenizer_and_vectorizer(self, config, seed, message_iterator_fn, message_count_fn):
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train_from_iterator(
            (message.text(config) for message in message_iterator_fn()),
            vocab_size=int(round(message_count_fn() * config['vocab_size_per_message'])),
        )

        seed_kwargs = {}
        if seed is not None:
            seed_kwargs['seed'] = seed
            seed_kwargs['workers'] = 1
        vectorizer = Doc2Vec(epochs=1, vector_size=config['document_vector_size'], min_count=config['vocab_token_min_count'], **seed_kwargs)

        frequencies = Counter()
        for message in tqdm(message_iterator_fn(), total=message_count_fn(), desc='Building vocabulary'):
            frequencies.update(tokenizer.encode(message.text(config)).tokens)
        vectorizer.build_vocab_from_freq(frequencies)

        count = message_count_fn()
        vectorizer.train(tqdm(iterable=(
            TaggedDocument(tokenizer.encode(message.text(config)).tokens, [message.uid])
            for message in message_iterator_fn()
        ), desc='Training doc2vec', total=count), total_examples=count, epochs=vectorizer.epochs)

        return tokenizer, vectorizer

    def _make_regressor(self, config, seed, vectors, labels):
        kwargs = {}
        prefix = config['regressor_type'] + '.'
        for key, value in config.items():
            if key.startswith(prefix):
                kwargs[key[len(prefix):]] = value

        regressor = REGRESSORS[config['regressor_type']](**kwargs)
        regressor.train(seed, vectors, labels)

        return regressor

    def _regressor_accuracy(self, regressor, vectors, labels):
        predictions = regressor.predict(vectors)
        predictions = np.clip(predictions, 0, 1)
        predictions[labels == False] = 1 - predictions[labels == False]
        return np.mean(predictions)

    def _regressor_brier_score(self, regressor, vectors, labels):
        predictions = regressor.predict(vectors)
        predictions = np.clip(predictions, 0, 1)
        return np.mean((predictions - labels) ** 2)

    def build(self, config=None):
        with TemporaryDirectory() as temp:
            temp = Path(temp)

            if config is None:
                config = CS.get_default_configuration()
            connector = CONNECTORS[self.settings.CONNECTOR](self.settings)

            with open(temp / 'config.json', 'w') as fp:
                json.dump(dict(config), fp)

            tokenizer, vectorizer = self._make_tokenizer_and_vectorizer(
                config, None,
                connector.iterate_all_messages,
                connector.estimate_total_message_count,
            )

            tokenizer.save(str(temp / 'tokenizer.json'))
            vectorizer.save(str(temp / 'doc2vec.model'))

            overestimate = int(round(connector.estimate_total_message_count() * 1.1))
            global_vectors = np.empty((overestimate, config['document_vector_size']), dtype=float)
            global_labels = np.empty(overestimate, dtype=bool)
            recipients = list(connector.recipients())
            idx = 0

            with tqdm(total=overestimate, desc='Preparing training data') as progress:
                for recipient in recipients:
                    recipient_start_idx = idx

                    for message in connector.iterate_messages_for_user(recipient):
                        if message.label is not None:
                            global_vectors[idx] = infer_vector(tokenizer, vectorizer, message.text(config))
                            global_labels[idx] = message.label
                            idx += 1
                        progress.update(1)

                    if len(recipients) > 1:
                        recipient_vectors = global_vectors[recipient_start_idx:idx]
                        recipient_labels = global_labels[recipient_start_idx:idx]

                        regressor = self._make_regressor(config, None, recipient_vectors, recipient_labels)

                        with open(temp / recipient / 'regressor.pkl', 'wb') as fp:
                            pickle.dump(regressor, fp)

                        score = self._regressor_accuracy(regressor, recipient_vectors, recipient_labels)
                        print(f'Training accuracy for <{recipient}>:', score)

                global_vectors = global_vectors[:idx]
                global_labels = global_labels[:idx]

                regressor = self._make_regressor(config, None, global_vectors, global_labels)

                with open(temp / 'regressor.pkl', 'wb') as fp:
                    pickle.dump(regressor, fp)

                score = self._regressor_accuracy(regressor, global_vectors, global_labels)
                print('Global training accuracy:', score)

            for item in self.settings.STORAGE.iterdir():
                if item.is_dir():
                    rmtree(item)
                else:
                    item.unlink()

            for item in temp.iterdir():
                move(item, self.settings.STORAGE / item.name)

    def initialize_hpo(self):
        connector = CONNECTORS[self.settings.CONNECTOR](self.settings)

        accessors = sorted(connector.iterate_all_message_accessors())
        Random('hpo').shuffle(accessors)

        split_index = int(round(len(accessors) * 0.1))
        self.validation_accessors = accessors[:split_index]
        self.training_accessors = accessors[split_index:]

        self.max_budget = len(self.training_accessors)
        self.min_budget = max(1, int(round(len(self.training_accessors) / 100)))

    def train_and_validate(self, config, seed, budget):
        budget = int(round(budget))
        connector = CONNECTORS[self.settings.CONNECTOR](self.settings)

        def train_message_count():
            return budget

        def train_message_iterator():
            yield from connector.fetch_messages_for_accessors(self.training_accessors[:budget])

        tokenizer, vectorizer = self._make_tokenizer_and_vectorizer(
            config, seed,
            train_message_iterator,
            train_message_count,
        )

        train_vectors = np.empty((budget, config['document_vector_size']), dtype=float)
        train_labels = np.empty(budget, dtype=bool)
        idx = 0

        for message in tqdm(train_message_iterator(), total=budget, desc='Converting training accessors'):
            if message.label is not None:
                train_vectors[idx] = infer_vector(tokenizer, vectorizer, message.text(config))
                train_labels[idx] = message.label
                idx += 1

        train_vectors = train_vectors[:idx]
        train_labels = train_labels[:idx]

        validation_length = len(self.validation_accessors)
        validation_vectors = np.empty((validation_length, config['document_vector_size']), dtype=float)
        validation_labels = np.empty(validation_length, dtype=bool)
        idx = 0

        for message in tqdm(connector.fetch_messages_for_accessors(self.validation_accessors), total=validation_length, desc='Converting validation accessors'):
            if message.label is not None:
                validation_vectors[idx] = infer_vector(tokenizer, vectorizer, message.text(config))
                validation_labels[idx] = message.label
                idx += 1

        validation_vectors = validation_vectors[:idx]
        validation_labels = validation_labels[:idx]

        regressor = self._make_regressor(config, seed, train_vectors, train_labels)
        return self._regressor_brier_score(regressor, validation_vectors, validation_labels)
