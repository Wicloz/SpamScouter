import Milter
from Milter import decode
from tokenizers import Tokenizer
from gensim.models.doc2vec import Doc2Vec
from argparse import ArgumentParser
from pathlib import Path
from .message import Message
import json
import email
import pickle


class SpamScouterMilter(Milter.Base):
    def __init__(self):
        self.recipients = []
        self.message = b''

    def envrcpt(self, address, *_):
        # store envelope recipients for loading models
        self.recipients.append(address)

        # tell the MTA to continue
        return Milter.CONTINUE

    @decode('bytes')
    def header(self, key, value):
        # add header to the reconstituted message
        self.message += key.encode('ascii')
        self.message += b': '
        self.message += value
        self.message += b'\r\n'

        # tell the MTA to continue
        return Milter.CONTINUE

    def eoh(self):
        # add a blank line to the reconstituted message
        self.message += b'\r\n'

        # tell the MTA to continue
        return Milter.CONTINUE

    def body(self, chunk):
        # add body chunk to the reconstituted message
        self.message += chunk

        # tell the MTA to continue
        return Milter.CONTINUE

    @staticmethod
    def _spam_probability(regressor, vector, previous=1):
        return min(max(regressor.predict([vector])[0], 0), previous)

    def eom(self):
        # do some logging
        print('Processing email for recipients:', self.recipients)

        # convert the message to text
        text = Message(email.message_from_bytes(self.message), None, None).text(CONFIG)
        print('>', len(text), 'characters after processing.')

        # convert the text to a vector
        vector = VECTORIZER.infer_vector(TOKENIZER.encode(text).tokens)

        # predict the global spam probability
        spam_probability = self._spam_probability(REGRESSOR, vector)

        # modify the spam probability for each recipient
        for recipient in self.recipients:
            if recipient in REGRESSORS:
                spam_probability = self._spam_probability(REGRESSORS[recipient], vector, spam_probability)

        # add the spam probability as a header
        print('> Spam Probability:', spam_probability)
        self.addheader('X-SpamScouter-Probability', f'{spam_probability:f}')

        # reset instance variables, more emails could be sent over this connection
        self.__init__()

        # tell the MTA to continue
        return Milter.CONTINUE


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_store_dir', type=Path)
    parser.add_argument('--protocol', type=str, default='inet')
    parser.add_argument('--address', type=str, default='3639@localhost')
    args = parser.parse_args()

    with open(args.model_store_dir / 'config.json', 'r') as fp:
        CONFIG = json.load(fp)

    TOKENIZER = Tokenizer.from_file(str(args.model_store_dir / 'tokenizer.json'))
    VECTORIZER = Doc2Vec.load(str(args.model_store_dir / 'doc2vec.model'))

    with open(args.model_store_dir / 'regressor.pkl', 'rb') as fp:
        REGRESSOR = pickle.load(fp)

    REGRESSORS = {}
    for pickle_file_path in args.model_store_dir.glob('*/regressor.pkl'):
        with open(pickle_file_path, 'rb') as fp:
            regressor = pickle.load(fp)
        REGRESSORS[pickle_file_path.parent.name] = regressor

    Milter.factory = SpamScouterMilter
    print('Done loading models, starting SpamScouter milter ...')
    Milter.runmilter('SpamScouterMilter', f'{args.protocol}:{args.address}', 60)
