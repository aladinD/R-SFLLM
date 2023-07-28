from datamodules.bert_datamodule import BertDataModule
import hydra
from models.bert_module import CustomBertModelModule
import pytorch_lightning as pl


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    model = CustomBertModelModule.from_pretrained(**cfg.model.config)
    datamodule = BertDataModule(**cfg.data)
    trainer = pl.Trainer(**cfg.trainer)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()