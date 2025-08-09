import Milter
from Milter import decode
from tokenizers import Tokenizer
from gensim.models.doc2vec import Doc2Vec
import torch
from argparse import ArgumentParser
from pathlib import Path
from .message import Message
from .models import SpamRegressor
import json
import email


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

    def eom(self):
        # do some logging
        print('Processing email for recipients:', self.recipients)

        # convert the message to text
        text = Message(email.message_from_bytes(self.message), None, None).text(CONFIG)
        print('>', len(text), 'characters after processing.')

        # convert the text to a tensor
        tensor = torch.tensor(list(VECTORIZER.infer_vector(TOKENIZER.encode(text).tokens)))

        # predict the global spam probability
        with torch.no_grad():
            spam_probability = REGRESSOR(tensor)['logits'].item()

        # modify the spam probability for each recipient
        for recipient in self.recipients:
            if recipient in REGRESSORS:
                with torch.no_grad():
                    spam_probability = min(spam_probability,
                                           REGRESSORS[recipient](tensor)['logits'].item())

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

    REGRESSOR = SpamRegressor(CONFIG)
    REGRESSOR.load(args.model_store_dir / 'regressor.pt')
    REGRESSOR.eval()

    REGRESSORS = {}
    for py_torch_file in args.model_store_dir.glob('*/regressor.pt'):
        regressor = SpamRegressor(CONFIG)
        regressor.load(py_torch_file)
        regressor.eval()
        REGRESSORS[py_torch_file.parent.name] = regressor

    Milter.factory = SpamScouterMilter
    print('Done loading models, starting SpamScouter milter ...')
    Milter.runmilter('SpamScouterMilter', f'{args.protocol}:{args.address}', 60)
