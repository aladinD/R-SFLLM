import copy
import random

import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from joblib import Parallel, delayed
from typing import Optional

from models.components.aggregator import Aggregator
from models.components.client import Client
from models.components.wireless_module import WirelessModule
import utils
from datamodules.qa_datamodule import SQUADDataModule
from models.bert_module import BERTForQuestionAnsweringModule
from utils import plotting 
from utils.utils import init_dir


# Seeding
seed = 42
pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)


def parallel_train(client, cfg, r):
    """
    Train and save a client model seperately in a parallel job.
    """
    logger = pl.loggers.CSVLogger(save_dir=cfg.training.client_log_path, name=f"client_{client.id}_logger", version=f"round_{r}")

    # Disable checkpoints logging 
    checkpoint_callback = ModelCheckpoint(save_top_k=0, every_n_epochs=int(1e9))
    
    client.trainer = pl.Trainer(**cfg.trainer, devices=[client.id], logger=logger, log_every_n_steps=1, callbacks=[checkpoint_callback])
    client.trainer.fit(client.model, client.train_data, client.val_data)
    torch.save(client.model.state_dict(), cfg.training.client_ckpts_path + f"client_{client.id}.pt")


def get_dls(cfg, master: bool = False): 
    """
    Loads the train and val dataloaders for each clients including appropriate splitting.
    Loads the complete train and val dataloaders for the master.
    """
    if master is False:
        datamodule = SQUADDataModule(**cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader()
    
    else:
        data_config = dict(cfg.data)
        data_config['num_splits'] = None

        datamodule = SQUADDataModule(**data_config)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader()


def evaluate_master_model(model, cfg, r):
    """
    Evaluates the master model on the complete train and validation dataset.
    """
    master_train_dl, master_val_dl = get_dls(cfg, master=True)

    master = Client(id=99,
                    model=copy.deepcopy(model),
                    trainer=None,
                    train_data=master_train_dl,
                    val_data=master_val_dl)
    
    master.model.load_state_dict(torch.load(cfg.training.master_ckpts_path + f"master_round_{r}.pt"))
    
    eval_config = dict(cfg.trainer)
    eval_config['devices'] = 1

    train_logger = pl.loggers.CSVLogger(save_dir=cfg.training.master_log_path, name="train", version=f"round_{r}")
    validation_logger = pl.loggers.CSVLogger(save_dir=cfg.training.master_log_path, name="validation", version=f"round_{r}")

    train_trainer = pl.Trainer(**eval_config, logger=train_logger, log_every_n_steps=1)
    train_trainer.test(master.model, master.train_data)

    validation_trainer = pl.Trainer(**eval_config, logger=validation_logger, log_every_n_steps=1)
    validation_trainer.test(master.model, master.val_data)


@hydra.main(version_base="1.3", config_path=".", config_name="config_qa")
def main(cfg):
    
    # Logger
    log = utils.get_pylogger(__name__)

    # Initialize directory
    log.info("INITIALIZING DIRECTORY")
    init_dir(cfg)

    # Get dataloaders
    log.info("LOADING DATA")
    train_dls, val_dls = get_dls(cfg, master=False)

    # Instantiate model
    log.info("INSTANTIATING MODEL")
    # if cfg.model.config.pretrained_model_name_or_path == "bert-base-uncased":
    #     model = BERTForTokenClassificationModule.from_pretrained(**cfg.model.config)
    # elif cfg.model.config.pretrained_model_name_or_path == "roberta-base":
    #     model = RoBERTaForTokenClassificationModule.from_pretrained(**cfg.model.config)
    # else:
    #     log.error("INVALID MODEL TYPE FROM : {bert-base-uncased, roberta-base}")

    model = BERTForQuestionAnsweringModule.from_pretrained(**cfg.model.config)

    # Set additional model configurations
    model.lr_val = cfg.optimizer.lr
    model.eps_val = cfg.optimizer.eps
    model.warmup = cfg.optimizer.warmup
    model.scheduler_training_steps = cfg.sfl.num_epochs * len(train_dls[0])
    model.num_classes = cfg.model.config.num_labels
    model.init_metrics()
    model.add_noise = False

    # Instantiate clients
    log.info("INSTANTIATING CLIENTS")
    clients = []
    for i in range(cfg.sfl.num_clients):
        client = Client(id=i,
                        model=copy.deepcopy(model),
                        trainer=pl.Trainer(**cfg.trainer, devices=[i]),
                        train_data=train_dls[i],
                        val_data=val_dls[i])
        clients.append(client)

    # Instantiate aggregator
    log.info("INSTANTIATING AGGREGATOR")
    aggregator = Aggregator(name="aggregator")

    # Instantiate wireless module
    wireless: Optional[WirelessModule] = hydra.utils.instantiate(cfg.wireless) if cfg.wireless is not None else cfg.wireless
    
    # SFL global round loop
    log.info("STARTING SFL TRAINING")
    for r in range(cfg.sfl.num_rounds):
        
        log.info(f"GLOBAL ROUND : {r+1} of {cfg.sfl.num_rounds}")
        # Simulate communication each round
        if wireless is not None:
            mses = wireless()
            log.info(f"Simulating comms scenario: {wireless.scenario}. MSEs: {mses}")
        # Client training loop
        if cfg.sfl.process == "sequential":

            # Load client models
            for i, client in enumerate(clients):
                # Update communication MSEs if needed
                if wireless is not None:
                    client.model.add_noise = mses[i]
                if r!= 0:
                    client.model.load_state_dict(torch.load(cfg.training.client_ckpts_path + f"client_{client.id}.pt"))
                else:
                    pass

                # Train client models sequentially on GPU:0
                logger = pl.loggers.CSVLogger(save_dir=cfg.training.client_log_path, name=f"client_{client.id}_logger", version=f"round_{r}")
                client.trainer = pl.Trainer(**cfg.trainer, devices=[0], logger=logger, log_every_n_steps=1) 
                client.trainer.fit(client.model, client.train_data, client.val_data)
                torch.save(client.model.state_dict(), cfg.training.clients_ckpts_path + f"client_{client.id}.pt")
                
        elif cfg.sfl.process == "parallel":
            # Load client models
            for i, client in enumerate(clients):
                # Update communication MSEs if needed
                if wireless is not None:
                    client.add_noise = mses[i]
                if r!= 0:
                    client.model.load_state_dict(torch.load(cfg.training.client_ckpts_path + f"client_{client.id}.pt"))
                else:
                    pass

            # Train client models in parallel
            Parallel(n_jobs=-1)(delayed(parallel_train)(client, cfg, r) for client in clients)

        else:
            log.error("INVALID PROCESS TYPE from {parallel, sequential}")

        log.info("ALL CLIENTS TRAINED")

        # Reload all clients to ensure proper model states after sequential/parallel training
        for client in clients:
            client.model.load_state_dict(torch.load(cfg.training.client_ckpts_path + f"client_{client.id}.pt"))

        # Aggregate client models
        attentions = aggregator.accumulate_attentions([client.model for client in clients])
        heads = aggregator.accumulate_heads([client.model for client in clients])
        embeddings = aggregator.accumulate_embeddings([client.model for client in clients])

        aggregated_attentions = aggregator.aggregate(attentions)
        aggregated_heads = aggregator.aggregate(heads)
        aggregated_embeddings = aggregator.aggregate(embeddings)

        # Model update & save
        for client in clients:
            client.update_model(aggregated_attentions)
            client.update_model(aggregated_heads)
            client.update_model(aggregated_embeddings)
            torch.save(client.model.state_dict(), cfg.training.client_ckpts_path + f"client_{client.id}.pt")

        log.info("ALL CLIENTS AGGREGATED")

        # Save master model
        torch.save(clients[-1].model.state_dict(), cfg.training.master_ckpts_path + f"master_round_{r}.pt")
        log.info("MASTER MODEL SAVED")

        # Evaluate master model
        log.info("EVALUATING MASTER MODEL")
        evaluate_master_model(model, cfg, r)

        # End round and save metrics plot
        if r == cfg.sfl.num_rounds - 1:
            log.info("ALL ROUNDS COMPLETED")
            # log.info("PLOTTING & SAVING METRICS")
            # plotting.plot_metrics(cfg, 
            #                       plot_name="result.png",
            #                       save_dir=cfg.training.plot_path, 
            #                       logs_path=cfg.training.log_path, 
            #                       plot_train_metrics=False)


if __name__ == "__main__":
    main()