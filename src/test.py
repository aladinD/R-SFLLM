from datamodules.bert_datamodule import BertDataModule
import hydra
from models.bert_module import CustomBertModelModule
import pytorch_lightning as pl


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    datamodule = BertDataModule(**cfg.data)

    # trainer = pl.Trainer(**cfg.trainer)
    # trainer.fit(model, datamodule=datamodule)

    # Testing Mutliple Instances
    datamodule.prepare_data()
    datamodule.setup()
    train_dls = datamodule.train_dataloader()
    val_dls = datamodule.val_dataloader()

    a = train_dls[0]
    print(len(a.dataset))

    b = train_dls[1]
    print(len(b.dataset))

    c = train_dls[2]
    print(len(c.dataset))

    # print(len(train_dls.dataset))

    # for i in range(3):
    #     model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    #     trainer = pl.Trainer(**cfg.trainer)
    #     trainer.fit(model, train_dls[i], val_dls[i])

    #     print(f"DONE TRAINING MODEL {i+1} of 3")


if __name__ == "__main__":
    main()