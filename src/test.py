from src.datamodules.bert_datamodule import BertDataModule
import hydra
from src.models.bert_module import CustomBertModelModule
import pytorch_lightning as pl
from aggregator import Aggregator
from client import Client
import torch.multiprocessing as mp


def parallel_train(client, cfg):
    """
    Wrapper function to train a client model seperately in a multiprocessing thread.
    """
    client.trainer = pl.Trainer(**cfg.trainer, devices=[client.id])
    client.trainer.fit(client.model, client.train_data, client.val_data)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    # Model and data module instantiations
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
        client = Client(id=i,
                        model=model,
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
        if cfg.sfl.process == "sequential":
            for client in clients:
                # Reset trainer to avoid max_epochs boundary
                client.trainer = pl.Trainer(**cfg.trainer, devices=[0])
                client.trainer.fit(client.model, client.train_data, client.val_data)
                
        elif cfg.sfl.process == "parallel":
            processes = []
            for client in clients:
                p = mp.Process(target=parallel_train, args=(client,cfg,))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        else:
            print("INVALID PROCESS TYPE")


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
            client.trainer.save_checkpoint(cfg.sfl.ckpt_path + f"{client.id}_model.ckpt")

            # Test
            # test_result = client.trainer.test(client.model, dataloaders=client.val_data)
            # print(f"Client {client.id} validation accuracy: {test_result[0]['test_acc']}")

        if r == cfg.sfl.num_rounds - 1:
            print("ALL ROUNDS COMPLETED")


if __name__ == "__main__":
    main()