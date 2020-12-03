from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network import NNClassifier
from loss import MyLoss


class NNClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Hyperparameters
        self.hparams = hparams

        # Classifier
        self.backbone = NNClassifier(input_size=self.hparams.input_dimentions,
                                     num_classes=self.hparams.num_classes)

        # Criterion
        self.criterion = MyLoss()

    def forward(self, inputs):
        return self.backbone(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, factor=self.hparams.lr_factor,
                                           patience=self.hparams.lr_patience, mode='min', verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': self.hparams.check_val_every_n_epoch
        }
        return [optimizer], [scheduler]

    def loss(self, logits, batch):
        return self.criterion(logits, batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='val')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # General
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=16)

        # Model
        parser.add_argument('--input_dimentions', type=int, default=100)
        parser.add_argument('--num_classes', type=int, default=30)

        # Loss
        parser.add_argument('--smoothing_eps', type=float, default=0.1)

        # Learning rate
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--lr_warmup_epochs', type=int, default=2)
        parser.add_argument('--lr_factor', type=float, default=0.1)
        parser.add_argument('--lr_patience', type=int, default=1)  # Two epochs with no improvement

        # Early stopping
        parser.add_argument('--es_patience', type=int, default=5)

        return parser

    def _step(self, batch, batch_idx, stage):
        logits = self.forward(batch['samples'])
        losses, metrics = self.loss(logits, batch['targets'])
        self._log(losses, metrics, stage=stage)
        return losses['total']

    def _log(self, losses, metrics, stage):
        pbar_prefix = 'train' if stage == 'train' else stage
        progress_bar = dict()
        progress_bar.update({f'{pbar_prefix}_acc': metrics['accuracy']})
        # progress_bar.update({f'{pbar_prefix}_loss': losses['total']})
        self.log_dict(progress_bar, prog_bar=True, logger=False)

        logs = dict()
        logs.update({f'losses/{stage}_{lname}': lval for lname, lval in losses.items()})
        logs.update({f'metrics/{stage}_{lname}': lval for lname, lval in metrics.items()})
        self.log_dict(logs, prog_bar=False, logger=True)

        if stage == 'val':
            # Log monitor
            self.log('val_loss', losses['total'])