from src.datamodules.bert_datamodule import BertDataModule
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
import time
from joblib import Parallel, delayed
import copy


def parallel_train(client, cfg, logger):
    """
    Wrapper function to train and save a client model seperately in a multiprocessing thread.
    """
    client.trainer = pl.Trainer(**cfg.trainer, devices=[client.id], logger=client.logger, log_every_n_steps=30)
    client.trainer.fit(client.model, client.train_data, client.val_data)
    torch.save(client.model.state_dict(), cfg.sfl.ckpt_path + f"client_{client.id}.pt")


def get_dls(cfg, master: bool = False): 
    """
    Loads the train and val dataloaders.
    """
    if master is False:
        datamodule = GLUEDataModule(**cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader()
    else:
        data_config = dict(cfg.data)
        data_config['num_splits'] = None
        datamodule = GLUEDataModule(**data_config)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.train_dataloader(), datamodule.val_dataloader()


def evaluate_master_model(master_model, cfg, master_train_logger, master_val_logger):
    """
    Evaluates the master model on the complete train and validation dataset.
    """
    master_train_dl, master_val_dl = get_dls(cfg, master=True)

    eval_config = dict(cfg.trainer)
    eval_config['devices'] = 1

    master_trainer = pl.Trainer(**eval_config, logger=master_train_logger)
    master_trainer.test(master_model, master_train_dl)

    master_trainer = pl.Trainer(**eval_config, logger=master_val_logger)
    master_trainer.test(master_model, master_val_dl)
    

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    
    # Loggers
    log = utils.get_pylogger(__name__)
    sfl_logger = pl_loggers.CSVLogger("logs", name="sfl_logger")
    # master_train_logger = pl_loggers.CSVLogger("logs", name="master_train_logger")
    # master_val_logger = pl_loggers.CSVLogger("logs", name="master_val_logger")

    # Get dataloaders
    train_dls, val_dls = get_dls(cfg, master=False)
    log.info("DATA LOADED WITH DATALOADER SIZE PER CLIENT : %s", len(train_dls[0]))

    # Instantiate clients 
    model = BERTModule.from_pretrained(**cfg.model.config)
    clients = []
    for i in range(cfg.sfl.num_clients):
        client = Client(id=i,
                        model=copy.deepcopy(model),
                        trainer=pl.Trainer(**cfg.trainer, devices=[i]),
                        logger=pl.loggers.CSVLogger("logs/clients", name=f"client_{i}_logger"),
                        train_data=train_dls[i],
                        val_data=val_dls[i])
        clients.append(client)

    # Instantiate aggregator
    aggregator = Aggregator(name="aggregator")

    # SFL global round loop
    for r in range(cfg.sfl.num_rounds):
        
        log.info(f"GLOBAL ROUND : {r+1} of {cfg.sfl.num_rounds}")

        # Train client models
        if cfg.sfl.process == "sequential":
            for client in clients:
                if r!= 0:
                    client.model.load_state_dict(torch.load(cfg.sfl.ckpt_path + f"client_{client.id}.pt"))
                else:
                    pass

                client.trainer = pl.Trainer(**cfg.trainer, devices=[3])
                client.trainer.fit(client.model, client.train_data, client.val_data)
                torch.save(client.model.state_dict(), cfg.sfl.ckpt_path + f"client_{client.id}.pt")
                
        elif cfg.sfl.process == "parallel":

            # Load client models
            for client in clients:
                if r!= 0:
                    client.model.load_state_dict(torch.load(cfg.sfl.ckpt_path + f"client_{client.id}.pt"))
                else:
                    pass

            Parallel(n_jobs=-1)(delayed(parallel_train)(client, cfg, sfl_logger) for client in clients)

        else:
            log.error("INVALID PROCESS TYPE from {parallel, sequential}")

        log.info("ALL CLIENTS TRAINED")

        # Reload all clients
        for client in clients:
            client.model.load_state_dict(torch.load(cfg.sfl.ckpt_path + f"client_{client.id}.pt"))

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
            torch.save(client.model.state_dict(), cfg.sfl.ckpt_path + f"client_{client.id}.pt")

        log.info("ALL CLIENTS AGGREGATED")

        # Save master model
        torch.save(clients[-1].model.state_dict(), cfg.sfl.master_path + f"master_round_{r}.pt")
        log.info("MASTER MODEL SAVED")

        # Evaluate master model
        log.info("EVALUATING MASTER MODEL")
        # evaluate_master_model(clients[-1].model, cfg, master_train_logger, master_val_logger)

        if r == cfg.sfl.num_rounds - 1:
            log.info("ALL ROUNDS COMPLETED")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()










# def parallel_train(client, cfg, logger):
#     """
#     Wrapper function to train and save a client model seperately in a multiprocessing thread.
#     """
#     client.trainer = pl.Trainer(**cfg.trainer, devices=[client.id], logger=logger)
#     client.trainer.fit(client.model, client.train_data, client.val_data)
#     torch.save(client.model.state_dict(), cfg.sfl.ckpt_path + f"client_{client.id}.pt")

# processes = []
# for client in clients:
#     p = mp.Process(target=parallel_train, args=(client,cfg,sfl_logger,))
#     processes.append(p)
#     p.start()

# for p in processes:
#     p.join()