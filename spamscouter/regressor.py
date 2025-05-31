import torch.nn as nn
from transformers import Trainer
from datasets import Dataset
import torch


class SpamRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

        self.fc1 = nn.Linear(200, 2000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vectors, labels=None):
        vectors = self.fc1(vectors)
        vectors = self.relu(vectors)
        vectors = self.fc2(vectors)
        vectors = self.sigmoid(vectors)

        result = {'logits': vectors}
        if labels is not None:
            result['loss'] = self.loss(vectors, labels)

        return result


def train_single_model(vectors, labels):
    model = SpamRegressor()
    dataset = Dataset.from_dict({'vectors': vectors, 'labels': labels})
    trainer = Trainer(model=model, train_dataset=dataset)
    trainer.train()
    return model


def train_all_models(path, connector, tokenizer, vectorizer):
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
            model = train_single_model(recipient_vectors, recipient_labels)
            torch.save(model.state_dict(), path / recipient / 'regressor.pt')

    model = train_single_model(global_vectors, global_labels)
    torch.save(model.state_dict(), path / 'regressor.pt')


def load_model_for_user(path, recipient=None):
    model = SpamRegressor()

    if recipient is not None:
        path = path / recipient

    model.load_state_dict(torch.load(path / 'regressor.pt'))
    model.eval()

    return model
