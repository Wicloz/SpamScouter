import Milter
from Milter import decode
from tokenizers import Tokenizer
from gensim.models.doc2vec import Doc2Vec
import torch
from argparse import ArgumentParser
from pathlib import Path
from .message import parse_milter_message
from .models import SpamRegressor


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
        # convert the message to text
        text = parse_milter_message(self.message)
        print(text)

        # convert the text to a vector
        vector = list(VECTORIZER.infer_vector(TOKENIZER.encode(text).tokens))
        print(vector)

        # predict the global spam probability
        with torch.no_grad():
            tensor = torch.tensor(vector, dtype=torch.float32).view(1, -1)
            spam_probability = REGRESSOR(tensor).item()

        # add the spam probability as a header
        print('Spam Probability:', spam_probability)
        self.addheader('X-Spam-Scouter-Probability', str(spam_probability))

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

    TOKENIZER = Tokenizer.from_file(str(args.model_store_dir / 'tokenizer.json'))
    VECTORIZER = Doc2Vec.load(str(args.model_store_dir / 'doc2vec.model'))

    REGRESSOR = SpamRegressor()
    REGRESSOR.load(args.model_store_dir / 'regressor.pt')
    REGRESSOR.eval()

    Milter.factory = SpamScouterMilter
    Milter.runmilter('SpamScouterMilter', f'{args.protocol}:{args.address}', 60)
