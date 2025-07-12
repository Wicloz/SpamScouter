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
import numpy as np


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


PT_DTYPE = torch.float32
NP_DTYPE = np.float32


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

            overestimate = int(round(connector.estimate_total_message_count() * 1.1))
            global_vectors = np.empty((overestimate, config['document_vector_size']), dtype=NP_DTYPE)
            global_labels = np.empty((overestimate, 1), dtype=NP_DTYPE)
            recipients = list(connector.recipients())
            idx = 0

            with tqdm(total=overestimate, desc='Preparing training data') as progress:
                for recipient in recipients:
                    recipient_start_idx = idx

                    for message in connector.iterate_messages_for_user(recipient):
                        if message.label is not None:
                            global_vectors[idx] = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                            global_labels[idx] = message.label
                            idx += 1
                        progress.update(1)

                    if len(recipients) > 1:
                        model = SpamRegressor(config)
                        model.fit(
                            torch.tensor(global_vectors[recipient_start_idx:idx], dtype=PT_DTYPE),
                            torch.tensor(global_labels[recipient_start_idx:idx], dtype=PT_DTYPE),
                        )
                        model.save(temp / recipient / 'regressor.pt')

                global_vectors = torch.tensor(global_vectors[:idx], dtype=PT_DTYPE)
                global_labels = torch.tensor(global_labels[:idx], dtype=PT_DTYPE)

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

        train_vectors = np.empty((budget, config['document_vector_size']), dtype=NP_DTYPE)
        train_labels = np.empty((budget, 1), dtype=NP_DTYPE)
        idx = 0

        for message in tqdm(train_message_iterator(), total=budget, desc='Converting training accessors'):
            if message.label is not None:
                train_vectors[idx] = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                train_labels[idx] = message.label
                idx += 1

        train_vectors = torch.tensor(train_vectors[:idx], dtype=PT_DTYPE)
        train_labels = torch.tensor(train_labels[:idx], dtype=PT_DTYPE)

        torch.manual_seed(seed)
        model = SpamRegressor(config)
        model.fit(train_vectors, train_labels)

        validation_length = len(self.validation_accessors)
        validation_vectors = np.empty((validation_length, config['document_vector_size']), dtype=NP_DTYPE)
        validation_labels = np.empty((validation_length, 1), dtype=NP_DTYPE)
        idx = 0

        for message in tqdm(connector.fetch_messages_for_accessors(self.validation_accessors), total=validation_length, desc='Converting validation accessors'):
            if message.label is not None:
                validation_vectors[idx] = vectorizer.infer_vector(tokenizer.encode(message.text(config)).tokens)
                validation_labels[idx] = message.label
                idx += 1

        validation_vectors = torch.tensor(validation_vectors[:idx], dtype=PT_DTYPE)
        validation_labels = torch.tensor(validation_labels[:idx], dtype=PT_DTYPE)

        return model.score(validation_vectors, validation_labels)
