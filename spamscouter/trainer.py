from .connectors.imap import ConnectorIMAP
from tokenizers import BertWordPieceTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import RULE_KEEP
from .models import SpamRegressor
from tempfile import TemporaryDirectory
from shutil import move, rmtree
from pathlib import Path


CONNECTORS = {
    'IMAP': ConnectorIMAP,
}


def train(config):
    with TemporaryDirectory() as temp:
        temp = Path(temp)

        connector = CONNECTORS[config.CONNECTOR](config)

        tokenizer = BertWordPieceTokenizer()
        tokenizer.train_from_iterator(message.text for message in connector.iterate_all_messages())
        tokenizer.save(str(temp / 'tokenizer.json'))

        vectorizer = Doc2Vec(vector_size=200)

        vectorizer.build_vocab(
            [TaggedDocument([key], [value]) for key, value in tokenizer.get_vocab().items()],
            trim_rule=lambda _1, _2, _3: RULE_KEEP,
        )

        for _ in range(2):
            vectorizer.train(
                (TaggedDocument(tokenizer.encode(message.text).tokens, [message.uid]) for message in connector.iterate_all_messages()),
                epochs=1, total_examples=connector.estimate_total_message_count(),
            )

        vectorizer.save(str(temp / 'doc2vec.model'))

        global_vectors = []
        global_labels = []

        recipients = list(connector.recipients())
        for recipient in recipients:
            recipient_vectors = []
            recipient_labels = []

            for message in connector.iterate_messages_for_user(recipient):
                if message.label is not None:
                    vector = vectorizer.infer_vector(tokenizer.encode(message.text).tokens)
                    recipient_vectors.append(vector)
                    recipient_labels.append([float(message.label)])
                    global_vectors.append(vector)
                    global_labels.append([float(message.label)])

            if len(recipients) > 1:
                model = SpamRegressor()
                model.fit(recipient_vectors, recipient_labels)
                model.save(temp / recipient / 'regressor.pt')

        model = SpamRegressor()
        model.fit(global_vectors, global_labels)
        model.save(temp / 'regressor.pt')

        for item in config.STORAGE.iterdir():
            if item.is_dir():
                rmtree(item)
            else:
                item.unlink()

        for item in temp.iterdir():
            move(item, config.STORAGE / item.name)
