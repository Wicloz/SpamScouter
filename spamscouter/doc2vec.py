from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import RULE_KEEP


def train_and_save_doc2vec(path, connector, tokenizer):
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

    vectorizer.save(str(path / 'doc2vec.model'))
    return vectorizer


def load_doc2vec(path):
    return Doc2Vec.load(str(path / 'doc2vec.model'))
