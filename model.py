import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import PointDataset, collate_fn
from typing import Tuple


class Tnet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.psi = nn.Sequential(nn.Conv1d(k, 64, 1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 128, 1),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 1024, 1),
                                 nn.BatchNorm1d(1024),
                                 nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, k**2),
                                 nn.ReLU())

    def forward(self, x) -> torch.Tensor:
        h1 = self.psi(x)
        h2 = torch.max(h1, dim=2)[0]
        mat = self.phi(h2)
        mat = mat.view(-1, self.k, self.k)
        out = torch.matmul(mat, x)
        return out


class PointNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.psi1 = nn.Sequential(Tnet(3),
                                  nn.Conv1d(3, 64, 1),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  Tnet(64)
                                  )
        self.psi2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1),
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 10),
                                 nn.Sigmoid())
        self.sigma = nn.Sequential(nn.Conv1d(1088, 512, 1),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(),
                                   nn.Conv1d(512, 256, 1),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Conv1d(256, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 2, 1),
                                   nn.BatchNorm1d(2),
                                   nn.Sigmoid(),
                                   )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.psi1(x)
        h2 = self.psi2(h1)
        h3 = torch.max(h2, dim=2)[0]
        y_hat = self.phi(h3)
        stack = torch.stack([h3] * h1.size(-1), dim=2)
        x_node = torch.concat([h1, stack], dim=1)
        y_hat_node = self.sigma(x_node)
        return y_hat, y_hat_node

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = torch.Tensor([0])
        for item in batch:
            x, y, y_node = item
            y_hat, y_hat_node = self.forward(x)
            loss += torch.nn.functional.binary_cross_entropy(y_hat, y)
            loss += torch.nn.functional.binary_cross_entropy(y_hat_node, y_node)
        loss /= len(batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    dataset = PointDataset('data/', noise=True)
    length = len(dataset)
    val_len = length // 10
    train_len = length - val_len
    train, val = random_split(dataset, [train_len, val_len])
    autoencoder = PointNet()
    trainer = pl.Trainer()
    trainer.fit(autoencoder, DataLoader(train, batch_size=32, collate_fn=collate_fn), DataLoader(val, batch_size=32,
                                                                                                 collate_fn=collate_fn))
