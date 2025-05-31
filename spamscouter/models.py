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


def train_regressor_for_user(vectors, labels):
    model = SpamRegressor()
    dataset = Dataset.from_dict({'vectors': vectors, 'labels': labels})
    trainer = Trainer(model=model, train_dataset=dataset)
    trainer.train()
    return model


def load_regressor_for_user(path, recipient=None):
    model = SpamRegressor()

    if recipient is not None:
        path = path / recipient

    model.load_state_dict(torch.load(path / 'regressor.pt'))
    model.eval()

    return model
