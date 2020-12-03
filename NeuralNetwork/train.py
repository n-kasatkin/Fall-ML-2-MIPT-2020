import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from module import NNClassifierModule
from datamodule import MyDataModule


warnings.filterwarnings("ignore", category=UserWarning)


def add_program_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General
    parser.add_argument('--experiment', type=str, default='baseline')

    # Paths
    parser.add_argument('--data_csv', type=str)
    parser.add_argument('--out_dir', type=str, default='output/')

    return parser


def config_args():
    parser = ArgumentParser()

    parser = add_program_specific_args(parser)
    parser = NNClassifierModule.add_model_specific_args(parser)
    parser = MyDataModule.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    return args


def checkpoint_callback(args):
    checkpoints_dir = f'{args.out_dir}/checkpoints/{args.experiment}'
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
    return ModelCheckpoint(
        filepath=checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min',
    )


def early_stopping_callback(args):
    return EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=True,
        patience=args.es_patience
    )


def tensorboard_logger(args):
    return TensorBoardLogger(
        save_dir=f'{args.out_dir}/logs',
        name=args.experiment
    )


def main(args):
    # Create data and model modules
    data = MyDataModule(args)
    model = NNClassifierModule(args)

    # Create trainer
    ckpt_callback = checkpoint_callback(args)
    es_callback = early_stopping_callback(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[ckpt_callback, es_callback],
        logger=tensorboard_logger(args),
    )

    # Fit
    trainer.fit(model, datamodule=data)

    # Verbose best model
    print(f'Best model score: {ckpt_callback.best_model_score:.3f}')
    print(f'Best model path: {ckpt_callback.best_model_path}')
    

if __name__ == "__main__":
    args = config_args()
    main(args)