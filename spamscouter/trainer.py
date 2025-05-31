from .tokenizer import train_and_save_tokenizer
from .doc2vec import train_and_save_doc2vec
from .regressor import train_all_models
from .connectors.imap import ConnectorIMAP


CONNECTORS = {
    'IMAP': ConnectorIMAP,
}


def train(config):
    connector = CONNECTORS[config.CONNECTOR](config)
    tokenizer = train_and_save_tokenizer(config.STORAGE, connector)
    vectorizer = train_and_save_doc2vec(config.STORAGE, connector, tokenizer)
    train_all_models(config.STORAGE, connector, tokenizer, vectorizer)
