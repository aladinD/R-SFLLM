import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import copy
import random
from typing import Optional, Union
from tqdm import tqdm
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning import LightningDataModule
import torch
from joblib import Parallel, delayed

from models.components.aggregator import Aggregator
from models.components.client import Client
from models.components.wireless_module import WirelessModule
import utils
from models.bert_module import BERTForTokenClassificationModule
from models.roberta_module import RoBERTaForTokenClassificationModule
from utils import plotting
from utils.utils import init_dir
from omegaconf import DictConfig, OmegaConf, open_dict


# Seeding
seed = 42
pl.seed_everything(seed, workers=True)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)


def train_single_client(client: Client, cfg: DictConfig, r: int, parallel: bool = True):
    """
    Train and save a client model seperately in a sequential or parallel job.
    """
    if parallel:
        cfg.trainer.devices = [client.id + 4]
    else:
        cfg.trainer.devices = [0]
        
    logger: CSVLogger = pl.loggers.CSVLogger(save_dir=cfg.paths.client_log_path, name=f"client_{client.id}_logger", version=f"round_{r}")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, log_every_n_steps=1)  
    client.trainer = trainer
    client.trainer.fit(client.model, client.train_data, client.val_data)
    torch.save(client.model.state_dict(), cfg.paths.client_ckpts_path + f"client_{client.id}.pt")


def get_dls(cfg: DictConfig, master: bool = False): 
    """
    Loads the train and val dataloaders for each clients including appropriate splitting.
    Loads the complete train and val dataloaders for the master.
    """
    if master is False:
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    else:
        data_config = cfg.datamodule # maybe deepcopy
        data_config.num_splits = None
        datamodule: LightningDataModule = hydra.utils.instantiate(data_config)

    datamodule.prepare_data()
    datamodule.setup()
    return datamodule.train_dataloader(), datamodule.val_dataloader()


def evaluate_master_model(model, cfg: DictConfig, r: int):
    """
    Evaluates the master model on the complete train and validation dataset.
    """
    master_train_dl, master_val_dl = get_dls(cfg, master=True)

    master = Client(id=99,
                    model=copy.deepcopy(model),
                    trainer=None,
                    train_data=master_train_dl,
                    val_data=master_val_dl)
    
    master.model.load_state_dict(torch.load(cfg.paths.master_ckpts_path + f"master_round_{r}.pt"))
    
    eval_config = copy.deepcopy(cfg.trainer)
    eval_config.devices = 1

    train_logger = pl.loggers.CSVLogger(save_dir=cfg.paths.master_log_path, name="train", version=f"round_{r}")
    validation_logger = pl.loggers.CSVLogger(save_dir=cfg.paths.master_log_path, name="validation", version=f"round_{r}")

    train_trainer: pl.Trainer = hydra.utils.instantiate(eval_config, logger=train_logger)
    train_trainer.test(master.model, master.train_data)

    validation_trainer: pl.Trainer = hydra.utils.instantiate(eval_config, logger=validation_logger)
    validation_trainer.test(master.model, master.val_data)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Logger
    log = utils.get_pylogger(__name__)

    # Initialize directory
    log.info(f"Initializing dirs in {cfg.paths.output_dir}")
    init_dir(cfg)

    # Get num_labels value
    with open_dict(cfg.datamodule):
        num_labels = cfg.datamodule.pop('num_labels')
    cfg.model.config.num_labels = num_labels

    # Instantiate model first, since model contains information needed for the dataloaders
    log.info(f"Instantiating model: {cfg.model._target_}")
    model_class: Union[BERTForTokenClassificationModule, RoBERTaForTokenClassificationModule] = hydra.utils.get_class(cfg.model._target_)
    model_cfg: DictConfig = cfg.model.config
    model: Union[BERTForTokenClassificationModule, RoBERTaForTokenClassificationModule] = model_class.from_pretrained(**model_cfg)

    # Get dataloaders
    log.info(f"Instantiating datamodule: {cfg.datamodule._target_}")
    cfg.datamodule.num_splits = cfg.sfl.num_clients
    cfg.datamodule.model_type = model_cfg.pretrained_model_name_or_path
    train_dls, val_dls = get_dls(cfg, master=False)
    print("LNE : ", len(train_dls[1]))

    # We defer setting additional model attributes, since LitModule does not allow for some reason 
    # to initialize this in the constructor and there is imo 
    # some very nasty coupling between model and datamodule which prevents this.
    model.lr_val = cfg.model.lr
    model.eps_val = cfg.model.eps
    model.warmup = cfg.model.warmup
    model.scheduler_training_steps = cfg.sfl.num_epochs * len(train_dls[0])
    model.num_classes = model_cfg.num_labels
    model.init_metrics()
    model.add_noise = False

    # Set the max epochs for training according to sfl!
    cfg.trainer.max_epochs = cfg.sfl.num_epochs

    # Pretty print stuff for debug and save config to file
    utils.extras(cfg=cfg)
    log.info(f"Instantiating Clients")
    # Instantiate clients
    clients = []
    client_configs = []
    for i in tqdm(range(cfg.sfl.num_clients)):
        client = Client(
            id=i,
            model=copy.deepcopy(model),
            trainer=None,
            train_data=train_dls[i],
            val_data=val_dls[i]
        )
        clients.append(client)
        conf_client = copy.deepcopy(cfg)
        OmegaConf.resolve(conf_client)
        client_configs.append(conf_client)

    # Instantiate aggregator
    log.info("INSTANTIATING AGGREGATOR")
    aggregator = Aggregator(name="aggregator")

    # Instantiate wireless module
    has_wireless = cfg.get("wireless", False)
    wireless: Optional[WirelessModule] = hydra.utils.instantiate(cfg.wireless) if has_wireless else None
    
    # SFL global round loop
    log.info("STARTING SFL TRAINING")
    for r in range(cfg.sfl.num_rounds):
        log.info(f"GLOBAL ROUND : {r+1} of {cfg.sfl.num_rounds}")
        # Simulate communication each round
        if wireless is not None:
            mses = wireless()
            log.info(f"Simulating comms scenario: {wireless.scenario}. MSEs: {mses}")
        # Client training loop
        # Load client models
        for i, client in enumerate(clients):
            # Update communication MSEs if needed
            if wireless is not None:
                client.model.add_noise = mses[i]
            if r!= 0:
                client.model.load_state_dict(torch.load(cfg.paths.client_ckpts_path + f"client_{client.id}.pt"))
            else:
                pass
        if cfg.sfl.process == "sequential":
            # Train client models sequentially on GPU:0
            for i, client in enumerate(clients):
                train_single_client(client=client, cfg=client_configs[i], r=r, parallel=False)
        elif cfg.sfl.process == "parallel":
            # Train client models in parallel
            Parallel(n_jobs=-1)(delayed(train_single_client)(client, conf, r, True) for (conf, client) in zip(client_configs, clients))
        else:
            raise TypeError("INVALID PROCESS TYPE from {parallel, sequential}")

        log.info("ALL CLIENTS TRAINED")

        # Reload all clients to ensure proper model states after sequential/parallel training
        for client in clients:
            client.model.load_state_dict(torch.load(cfg.paths.client_ckpts_path + f"client_{client.id}.pt"))

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
            torch.save(client.model.state_dict(), cfg.paths.client_ckpts_path + f"client_{client.id}.pt")

        log.info("ALL CLIENTS AGGREGATED")

        # Save master model
        torch.save(clients[-1].model.state_dict(), cfg.paths.master_ckpts_path + f"master_round_{r}.pt")
        log.info("MASTER MODEL SAVED")

        # Evaluate master model
        log.info("EVALUATING MASTER MODEL")
        evaluate_master_model(model, cfg, r)

        # End round and save metrics plot
        if r == cfg.sfl.num_rounds - 1:
            log.info("ALL ROUNDS COMPLETED")
            log.info("PLOTTING & SAVING METRICS")
            plotting.plot_metrics(
                cfg, 
                plot_name="result.png",
                save_dir=cfg.paths.plot_path, 
                logs_path=cfg.paths.log_path, 
                plot_train_metrics=False
            )


if __name__ == "__main__":
    main()