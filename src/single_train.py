from src.datamodules.glue_datamodule import GLUEDataModule
import hydra
from src.models.bert_module import BERTModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from sfl.aggregator import Aggregator
from sfl.client import Client
import torch.multiprocessing as mp
from src import utils
import torch
import copy
import uuid
from pytorch_lightning.utilities.parsing import AttributeDict

import torch
import random
import numpy as np

# Set random seed for PyTorch
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set random seed for Python's built-in random module
random.seed(seed)

# Set random seed for NumPy
np.random.seed(seed)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):

    # Get dataloaders
    datamodule = GLUEDataModule(**cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    print(len(train_dl[0]))

    # Get model
    model = BERTModule.from_pretrained(**cfg.model.config)
    client = Client(id=0,
                    model=model,
                    trainer=pl.Trainer(**cfg.trainer, devices=[0]),
                    logger=pl.loggers.CSVLogger("logs", name=f"test_log"),
                    train_data=train_dl[0],
                    val_data=val_dl[0])

    # Train model
    client.trainer = pl.Trainer(**cfg.trainer, devices=[0], logger=client.logger)
    client.trainer.fit(client.model, client.train_data, client.val_data)

    # Evaluate model
    client.trainer.test(client.model, client.val_data)


if __name__ == "__main__":
    main()
