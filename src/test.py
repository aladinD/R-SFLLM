from datamodules.bert_datamodule import BertDataModule
import hydra
from models.bert_module import CustomBertModelModule
import pytorch_lightning as pl
from aggregator import Aggregator


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

    # SFL global round loop
    for r in range(cfg.sfl.num_rounds):
        
        print(f"GLOBAL ROUND : {r+1} of {num_rounds}")

        # Instantiate clients
        for i in range(cfg.sfl.num_clients):
            clients.append(CustomBertModelModule.from_pretrained(**cfg.model.config))










    # # Module instantiation
    # model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    # datamodule = BertDataModule(**cfg.data)

    # # Data split
    # datamodule.prepare_data()
    # datamodule.setup()
    # train_dls = datamodule.train_dataloader()
    # val_dls = datamodule.val_dataloader()

    # # Testing multiple instances
    # for i in range(3):
    #     model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    #     trainer = pl.Trainer(**cfg.trainer)
    #     trainer.fit(model, train_dls[i], val_dls[i])

    #     for name, param in model.named_parameters():
    #         print(name, param)

    #     print(f"DONE TRAINING MODEL {i+1} of 3")





























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