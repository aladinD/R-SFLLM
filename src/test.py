from datamodules.bert_datamodule import BertDataModule
import hydra
from models.bert_module import CustomBertModelModule
import pytorch_lightning as pl
from aggregator import Aggregator
from client import Client


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    # Module instantiation
    model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    datamodule = BertDataModule(**cfg.data)

    # Data split
    datamodule.prepare_data()
    datamodule.setup()
    train_dls = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()

    # Instantiate clients 
    clients = []
    for i in range(cfg.sfl.num_clients):
        client = Client(name=f"client_{i+1}",
                        model=model if i == 0 else model.from_pretrained(**cfg.model.config),
                        trainer=pl.Trainer(**cfg.trainer),
                        train_data=train_dls[i],
                        val_data=val_dls[i])
        clients.append(client)

    # Instantiate aggregator
    aggregator = Aggregator(name="sfl_aggregator")

    # SFL global round loop
    for r in range(cfg.sfl.num_rounds):
        
        print(f"GLOBAL ROUND : {r+1} of {cfg.sfl.num_rounds}")

        # Train client models
        for client in clients:
            # Reset trainer to avoid max_epochs boundary
            client.trainer = pl.Trainer(**cfg.trainer)
            client.trainer.fit(client.model, client.train_data, client.val_data)

        print("ALL CLIENTS TRAINED")

        # Aggregate client models
        attentions = aggregator.accumulate_attentions([client.model for client in clients])
        heads = aggregator.accumulate_heads([client.model for client in clients])
        embeddings = aggregator.accumulate_embeddings([client.model for client in clients])

        aggregated_attentions = aggregator.aggregate(attentions)
        aggregated_heads = aggregator.aggregate(heads)
        aggregated_embeddings = aggregator.aggregate(embeddings)

        # Model update and save
        for client in clients:
            client.update_model(aggregated_attentions)
            client.update_model(aggregated_heads)
            client.update_model(aggregated_embeddings)

        if r == cfg.sfl.num_rounds - 1:
            print("ALL ROUNDS COMPLETED")
            

# @hydra.main(version_base="1.3", config_path=".", config_name="config")
# def main(cfg):
#     # trainer = pl.Trainer(**cfg.trainer)
#     # trainer.fit(model, datamodule=datamodule)

#     # Module instantiation
#     model = CustomBertModelModule.from_pretrained(**cfg.model.config)
#     datamodule = BertDataModule(**cfg.data)

#     # Data split
#     datamodule.prepare_data()
#     datamodule.setup()
#     train_dls = datamodule.train_dataloader()
#     val_dls = datamodule.val_dataloader()

#     # Testing multiple instances
#     for i in range(3):
#         model = CustomBertModelModule.from_pretrained(**cfg.model.config)
#         trainer = pl.Trainer(**cfg.trainer)
#         trainer.fit(model, train_dls[i], val_dls[i])

#         print(f"DONE TRAINING MODEL {i+1} of 3")


if __name__ == "__main__":
    main()