from datasets import load_dataset
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class BertDataModule(LightningDataModule):
    def __init__(self, 
                 glue_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 truncate: int = None)  -> None:
        super().__init__()
        self.glue_dataset = glue_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.truncate = truncate


    def prepare_data(self) -> None:
        self.dataset = load_dataset('glue', self.glue_dataset)


    def setup(self, stage: str = None) -> None:
        # Train/val split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self.train_dataset.select(range(self.truncate))
            self.val_dataset = self.val_dataset.select(range(self.truncate))
        else:
            pass

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)

        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tokenizer = BertTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(dataset["sentence"], truncation=True, padding=True)
        labels = torch.tensor(dataset["label"], dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        return dataset  