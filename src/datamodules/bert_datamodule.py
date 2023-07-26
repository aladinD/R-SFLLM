from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class BertDataModule(LightningDataModule):
    def __init__(self, 
                 glue_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.glue_dataset = glue_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_splits = num_splits
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

        # Splitting
        if self.num_splits is not None:
            self.train_dataset = self._split_data(self.train_dataset, self.num_splits)[0]
            self.val_dataset = self._split_data(self.val_dataset, self.num_splits)[0]
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
        labels = dataset["label"]
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        return dataset 


    def _split_data(self, 
                    dataset: DatasetDict, 
                    num_splits: int) -> None:
        """
        Splits the dataset into equal num_splits splits.
        """
        return torch.utils.data.random_split(dataset, num_splits)