from tokenizers import BertWordPieceTokenizer


def train_and_save_tokenizer(path, connector):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train_from_iterator(message.text for message in connector.iterate_all_messages())

    tokenizer.save(str(path / 'tokenizer.json'))
    return tokenizer


def load_tokenizer(path):
    return BertWordPieceTokenizer(str(path / 'tokenizer.json'))
