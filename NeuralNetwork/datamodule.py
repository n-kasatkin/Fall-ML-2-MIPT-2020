import os
import os.path as osp
from argparse import ArgumentParser
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MyDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Transform placeholders
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Split items
        X, y, features = MyDataset.load_data(self.hparams.data_csv)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.hparams.val_size)

        # Create train dataset
        self.train_dataset = MyDataset(X_train, y_train)

        # Create val dataset
        self.val_dataset = MyDataset(X_val, y_val)

    def train_dataloader(self):
        params = {'batch_size': self.hparams.batch_size,
                  'drop_last': False,
                  'shuffle': True,
                  'pin_memory': True,
                  'num_workers': self.hparams.num_workers}

        return DataLoader(self.train_dataset, **params)

    def val_dataloader(self):
        params = {'batch_size': self.hparams.batch_size,
                  'drop_last': False,
                  'shuffle': False,
                  'pin_memory': True,
                  'num_workers': self.hparams.num_workers}

        return DataLoader(self.val_dataset, **params)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--val_size', type=float, default=0.1)
        parser.add_argument('--train_steps_per_epoch', type=int, default=None)
        parser.add_argument('--val_steps_per_epoch', type=int, default=None)

        return parser