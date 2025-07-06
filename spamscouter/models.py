import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import torch
import numpy as np
from tempfile import TemporaryDirectory


class SpamRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = nn.BCELoss()

        self.fc1 = nn.Linear(config['doc2vec_output_size'], config['hidden_layer_size'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['hidden_layer_size'], 1)
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

    def score(self, vectors, labels):
        vectors = torch.tensor(np.array(vectors, dtype=np.float32), dtype=torch.float32)
        labels = torch.tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)

        with torch.no_grad():
            return self.forward(vectors, labels)['loss'].item()

    def fit(self, vectors, labels, seed=None):
        train_eval_split_idx = int(round(len(vectors) * 0.1))
        eval_dataset = Dataset.from_dict({'vectors': vectors[:train_eval_split_idx], 'labels': labels[:train_eval_split_idx]})
        train_dataset = Dataset.from_dict({'vectors': vectors[train_eval_split_idx:], 'labels': labels[train_eval_split_idx:]})

        seed_kwargs = {}
        if seed is not None:
            seed_kwargs['seed'] = seed
            seed_kwargs['full_determinism'] = True

        with TemporaryDirectory() as tmpdir:
            Trainer(model=self, train_dataset=train_dataset, eval_dataset=eval_dataset, args=TrainingArguments(
                **seed_kwargs,
                num_train_epochs=300,
                per_device_train_batch_size=int(round(len(vectors) / 100)),
                metric_for_best_model='loss',
                eval_strategy='epoch',
                per_device_eval_batch_size=int(round(len(vectors) / 100)),
                load_best_model_at_end=True,
                save_strategy='epoch',
                output_dir=tmpdir,
            ), callbacks=[EarlyStoppingCallback(
                early_stopping_patience=3,
            )]).train()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
