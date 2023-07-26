import pytorch_lightning as pl

from models.bert_module import CustomBertModel
from datamodules.bert_datamodule import BertDataModule

def main():
    model = CustomBertModel.from_pretrained("bert-base-uncased", num_labels=2)
    datamodule = BertDataModule(glue_dataset="sst2", batch_size=32, num_workers=1, truncate=100)
    # train_data, val_data = datamodule.train_dataloader(), datamodule.val_dataloader()
    trainer = pl.Trainer()
    trainer.fit(model, datamodule= datamodule)


if __name__ == "__main__":
    main()
