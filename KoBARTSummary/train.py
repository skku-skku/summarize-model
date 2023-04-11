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

class KoBARTConditionalGenerationTrainer():
    def __init__(self, args) -> None:
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.args = args
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

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

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join('tb_logs'))
        lr_logger = pl.callbacks.LearningRateMonitor()
        trainer = pl.Trainer.from_argparse_self.args(self.args, logger=tb_logger,
                                                callbacks=[checkpoint_callback, lr_logger])

        model = KoBARTConditionalGeneration(self.args, trainer)
        trainer.fit(model, dm)

        train_model.model.eval()

        torch.save({
            'model_state_dict': train_model.state_dict()
            }, 'output.pth')
