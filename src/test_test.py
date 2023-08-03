from src.datamodules.bert_datamodule import BertDataModule
import hydra
from src.models.bert_module import CustomBertModelModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from .aggregator import Aggregator
from .client import Client
import torch.multiprocessing as mp
from src import utils
import torch
import copy
import uuid
from pytorch_lightning.utilities.parsing import AttributeDict


def parallel_train(client, trainer_config):
    """
    Wrapper function to train and save a client model seperately in a multiprocessing thread.
    """
    client.trainer = pl.Trainer(**trainer_config, devices=[client.id])
    client.trainer.fit(client.model, client.train_data, client.val_data)

    
def test_parallel_accumulation():
    """
    Tests the parallel training and subsequent model parameter accumulation. 
    """

    # Hyperparameters
    num_clients = 2
    num_rounds = 2
    num_epochs = 5

    # Sample configs
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })

    data_config = AttributeDict({
        "glue_dataset": "sst2",
        "batch_size": 32,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "num_splits": num_clients,
        "truncate": 100
    })
        
    trainer_config = AttributeDict({
        "min_epochs": 1,
        "max_epochs": num_epochs,
        "accelerator": "cuda",
    })

    # Sample dataloaders
    datamodule = BertDataModule(**data_config)
    datamodule.prepare_data()
    datamodule.setup()
    train_dls = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()
                        
    # Sample clients
    model = CustomBertModelModule.from_pretrained(**model_config)
    clients = []
    for i in range(num_clients):
        client = Client(id=i,
                        model=model,
                        trainer=pl.Trainer(**trainer_config),
                        train_data=train_dls[i],
                        val_data=val_dls[i])
        clients.append(client)

    # Sample aggregator
    aggregator = Aggregator(name="test_aggregator")

    # Test 1: Both models should be equal before training
    model1 = clients[0].model
    model2 = clients[1].model

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1.startswith("bert.encoder") or name1.startswith("classifier") or name1.startswith("bert.pooler") or name1.startswith("bert.embeddings"):
            assert torch.any(param1 == param2) 

    # Global training
    for r in range(num_rounds):
        processes = []
        for client in clients:
            p = mp.Process(target=parallel_train, args=(client,trainer_config,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
    # Both models should not be equal after training
    # model1 = clients[0].model
    # model2 = clients[1].model

    # for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    #     if name1.startswith("bert.encoder") or name1.startswith("classifier") or name1.startswith("bert.pooler") or name1.startswith("bert.embeddings"):
    #         assert torch.any(param1 != param2)        

    # Aggregate client models
    attentions = aggregator.accumulate_attentions([client.model for client in clients])
    heads = aggregator.accumulate_heads([client.model for client in clients])
    embeddings = aggregator.accumulate_embeddings([client.model for client in clients])

    aggregated_attentions = aggregator.aggregate(attentions)
    aggregated_heads = aggregator.aggregate(heads)
    aggregated_embeddings = aggregator.aggregate(embeddings)

    # Model update
    for client in clients:
        client.update_model(aggregated_attentions)
        client.update_model(aggregated_heads)
        client.update_model(aggregated_embeddings)

    # Both models should be equal after aggregation
    model1 = clients[0].model
    model2 = clients[1].model

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1.startswith("bert.encoder") or name1.startswith("classifier") or name1.startswith("bert.pooler") or name1.startswith("bert.embeddings"):
            assert torch.all(param1 == param2)