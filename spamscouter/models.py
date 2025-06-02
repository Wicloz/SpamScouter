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

    def fit(self, vectors, labels):
        dataset = Dataset.from_dict({'vectors': vectors, 'labels': labels})
        trainer = Trainer(model=self, train_dataset=dataset)
        trainer.train()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
