import torch.nn as nn
import torch
import numpy as np
from tempfile import TemporaryDirectory
from tqdm import tqdm, trange
from math import ceil


class SpamRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = nn.BCELoss()

        self.fc1 = nn.Linear(config['document_vector_size'], config['hidden_layer_size'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['hidden_layer_size'], 1)
        self.sigmoid = nn.Sigmoid()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

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
        vectors = vectors.to(self.device)
        labels = labels.to(self.device)

        self.eval()
        with torch.no_grad():
            return self.forward(vectors, labels)['loss'].item()

    def fit(self, vectors, labels):
        train_eval_split_idx = int(round(len(vectors) * 0.1))
        train_eval_perm = torch.randperm(len(vectors))

        vectors = vectors[train_eval_perm].to(self.device)
        valid_vectors = vectors[:train_eval_split_idx]
        train_vectors = vectors[train_eval_split_idx:]

        labels = labels[train_eval_perm].to(self.device)
        valid_labels = labels[:train_eval_split_idx]
        train_labels = labels[train_eval_split_idx:]

        optimizer = torch.optim.Adam(self.parameters())
        incumbent_loss = float('inf')
        epochs_without_new_incumbent = 0

        with TemporaryDirectory() as tmpdir:
            with tqdm(desc='classifier Epochs') as progress:
                while True:
                    self.train()

                    epoch_perm = torch.randperm(len(train_vectors))
                    steps = ceil(len(train_vectors) / 100)
                    deficit = 100 - (len(train_vectors) % 100)
                    deficit_per_step = ceil(deficit / steps)
                    idx = 0

                    for _ in trange(steps, leave=False):
                        deficit_consumed = min(deficit, deficit_per_step)
                        batch_size = 100 - deficit_consumed
                        deficit -= deficit_consumed
                        batch_indices = epoch_perm[idx:idx + batch_size]
                        idx += batch_size

                        optimizer.zero_grad()
                        loss = self.forward(train_vectors[batch_indices], train_labels[batch_indices])['loss']
                        loss.backward()
                        optimizer.step()

                    progress.update(1)
                    self.eval()

                    with torch.no_grad():
                        eval_loss = self.forward(valid_vectors, valid_labels)['loss'].item()

                    if eval_loss < incumbent_loss:
                        incumbent_loss = eval_loss
                        epochs_without_new_incumbent = 0
                        self.save(f'{tmpdir}/incumbent.pt')

                    else:
                        epochs_without_new_incumbent += 1
                        if epochs_without_new_incumbent >= 10:
                            break

            self.load(f'{tmpdir}/incumbent.pt')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
