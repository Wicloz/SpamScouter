from .connectors.imap import ConnectorIMAP
from .connectors.cache import ConnectorCache
from tokenizers import BertWordPieceTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
from .models import SpamRegressor
from tempfile import TemporaryDirectory
from shutil import move, rmtree
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, Beta
from random import shuffle
from .message import MESSAGE_PROCESS_METHODS
import json
from tqdm import trange, tqdm
import torch


CONNECTORS = {
    'IMAP': ConnectorIMAP,
    'CACHE': ConnectorCache,
}


CS = ConfigurationSpace()
CS.add(Integer('document_vector_size', (100, 1000), default=200))
CS.add(Integer('hidden_layer_size', (100, 10000), default=2000))
CS.add(Categorical('message_processing_method', MESSAGE_PROCESS_METHODS.keys(), default='body_unicode'))
CS.add(Float('vocab_size_per_message', (0, 2), default=1, distribution=Beta(4, 4)))
CS.add(Integer('vocab_token_min_count', (1, 10000), default=1, log=True))


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
        vectorizer = Doc2Vec(vector_size=config['document_vector_size'], min_count=config['vocab_token_min_count'], **seed_kwargs)

        frequencies = Counter()
        for message in tqdm(message_iterator_fn(), total=message_count_fn(), desc='Building vocabulary'):
            frequencies.update(tokenizer.encode(message.text(config)).tokens)
        vectorizer.build_vocab_from_freq(frequencies)

        for _ in trange(3, desc='doc2vec Epochs'):
            count = message_count_fn()
            vectorizer.train(
                tqdm(iterable=(TaggedDocument(tokenizer.encode(message.text(config)).tokens, [message.uid]) for message in message_iterator_fn()), total=count),
                epochs=1, total_examples=count,
            )

        return tokenizer, vectorizer

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

            global_vectors = []
            global_labels = []

            recipients = list(connector.recipients())
            for recipient in recipients:
                recipient_vectors = []
                recipient_labels = []

                for message in connector.iterate_messages_for_user(recipient):
                    if message.label is not None:
                        vector = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                        recipient_vectors.append(vector)
                        recipient_labels.append([float(message.label)])
                        global_vectors.append(vector)
                        global_labels.append([float(message.label)])

                if len(recipients) > 1:
                    model = SpamRegressor(config)
                    model.fit(recipient_vectors, recipient_labels)
                    model.save(temp / recipient / 'regressor.pt')

            model = SpamRegressor(config)
            model.fit(global_vectors, global_labels)
            model.save(temp / 'regressor.pt')

            for item in self.settings.STORAGE.iterdir():
                if item.is_dir():
                    rmtree(item)
                else:
                    item.unlink()

            for item in temp.iterdir():
                move(item, self.settings.STORAGE / item.name)

    def initialize_hpo(self):
        connector = CONNECTORS[self.settings.CONNECTOR](self.settings)
        accessors = list(connector.iterate_all_message_accessors())
        shuffle(accessors)

        split_index = int(round(len(accessors) * 0.1))
        self.validation_accessors = accessors[:split_index]
        self.training_accessors = accessors[split_index:]

        self.max_budget = len(self.training_accessors)
        self.min_budget = max(1, int(round(len(self.training_accessors) / 100)))

    def train_and_validate(self, config, seed, budget, fast=False):
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

        train_vectors = []
        train_labels = []
        for message in train_message_iterator():
            if message.label is not None:
                vector = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                train_vectors.append(vector)
                train_labels.append([float(message.label)])

        torch.manual_seed(seed)
        model = SpamRegressor(config)
        model.fit(train_vectors, train_labels)

        validation_vectors = []
        validation_labels = []
        for message in connector.fetch_messages_for_accessors(self.validation_accessors if not fast else self.validation_accessors[:budget]):
            if message.label is not None:
                vector = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                validation_vectors.append(vector)
                validation_labels.append([float(message.label)])

        return model.score(validation_vectors, validation_labels)
