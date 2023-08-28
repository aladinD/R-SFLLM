from src.datamodules.bert_datamodule import BertDataModule
import hydra
from src.models.bert_module import CustomBertModelModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from aggregator import Aggregator
from client import Client
import torch.multiprocessing as mp
from src import utils
import torch
import copy
import uuid
from pytorch_lightning.utilities.parsing import AttributeDict

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):

    # Get dataloaders
    datamodule = BertDataModule(**cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    print(len(train_dl[0]))

    # Get model
    model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    client = Client(id=0,
                    model=model,
                    trainer=pl.Trainer(**cfg.trainer, devices=[0]),
                    train_data=train_dl[0],
                    val_data=val_dl[0])

    # Train model
    client.trainer.fit(client.model, client.train_data, client.val_data)

    # Evaluate model
    client.trainer.test(client.model, client.val_data)


if __name__ == "__main__":
    main()
