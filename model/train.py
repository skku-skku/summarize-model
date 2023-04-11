import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KobartSummaryModule
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from .model import Base, KoBARTConditionalGeneration

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='../data/train.csv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='../data/test.csv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class KoBARTConditionalGenerationTrainer():
    def __init__(self) -> None:
        parser = Base.add_model_specific_args(parser)
        parser = ArgsBase.add_model_specific_args(parser)
        parser = KobartSummaryModule.add_model_specific_args(parser)
        parser = pl.Trainer.add_argparse_args(parser)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.args = parser.parse_args()

    def train(self):
        logging.info(self.args)
        print(self.args)

        train_model = KoBARTConditionalGeneration(self.args)

        dm = KobartSummaryModule(self.args.train_file,
                            self.args.test_file,
                            self.tokenizer,
                            batch_size=self.args.batch_size,
                            max_len=self.args.max_len,
                            num_workers=self.args.num_workers)
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                        dirpath=self.args.default_root_dir,
                                                        filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                        verbose=True,
                                                        save_last=True,
                                                        mode='min',
                                                        save_top_k=3)

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(self.args.default_root_dir, 'tb_logs'))
        lr_logger = pl.callbacks.LearningRateMonitor()
        trainer = pl.Trainer.from_argparse_self.args(self.args, logger=tb_logger,
                                                callbacks=[checkpoint_callback, lr_logger])

        model = KoBARTConditionalGeneration(self.args, trainer)
        trainer.fit(model, dm)

        train_model.model.eval()

        torch.save({
            'model_state_dict': train_model.state_dict()
            }, 'output.pth')
