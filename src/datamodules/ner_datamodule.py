from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List
from abc import ABC, abstractmethod
import os


class NERDataModuleBase(LightningDataModule, ABC):
    """
    Abstract Base Class for NER Data Modules.
    """
    def __init__(self, 
                 ner_dataset: str, 
                 batch_size: int,
                 num_workers: int,
                 model_type: str,
                 num_splits: int = None,
                 truncate: int = None)  -> None:
        super().__init__()
        self.ner_dataset = ner_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.num_splits = num_splits
        self.truncate = truncate

        # Disable warnings when using AutoTokenizer instead of explicit tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def prepare_data(self) -> None:
        """"
        Loads the specified NER dataset.
        """
        if self.ner_dataset == "conll2012_ontonotesv5":
            self.dataset = load_dataset(self.ner_dataset, "english_v12")
        else:
            self.dataset = load_dataset(self.ner_dataset)


    @abstractmethod
    def setup(self, stage: str = None) -> None:
        """
        Abstract method for preprocessing the loaded dataset.
        """
        pass


    def train_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the training dataset.
        """
        if self.num_splits is not None:
            splits = self._split_data(self.train_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        """
        Returns a Dataloader object for the validation dataset.
        """
        if self.num_splits is not None:
            splits = self._split_data(self.val_dataset, self.num_splits)
            return [DataLoader(split, batch_size=self.batch_size) for split in splits]
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    @abstractmethod
    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Abstract method for tokenizing the dataset.
        """
        pass


    def _truncate(self, dataset) -> DatasetDict:
        """
        Truncates the dataset to the specified length.
        """
        return dataset.select(range(self.truncate))


    def _split_data(self, 
                    dataset: DatasetDict, 
                    num_splits: int) -> List[DatasetDict]:
        """
        Splits the dataset into equal num_splits splits.
        """
        split_size = len(dataset) // num_splits
        remainder = len(dataset) % num_splits

        split_lengths = [split_size + 1 if i < remainder else split_size for i in range(num_splits)]
        splits = torch.utils.data.random_split(dataset, split_lengths)

        return splits


class CoNLL2003DataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the CoNLL-2003 dataset. 
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tag_values = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'PAD']
        tag2idx = {t: i for i, t in enumerate(tag_values)}

        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        sentences = [" ".join(tokens) for tokens in dataset["tokens"]]
        encodings = tokenizer(sentences, truncation=True, padding=True)

        labels = []
        for i, label_seq in enumerate(dataset["ner_tags"]):
            label_seq = [tag2idx[tag_values[label]] for label in label_seq]
            padding_length = len(encodings['input_ids'][i]) - len(label_seq)
            labels.append(label_seq + ([tag2idx['PAD']] * padding_length))

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset
    

class WNUT17DataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the WNUT-17 dataset. 
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val/test split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        self.train_dataset = self._tokenize(self.train_dataset)
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tag_values = ['O', 'B-corporation', 'I-corporation', 'B-creative-work', 'I-creative-work', 'B-group', 
                           'I-group', 'B-location', 'I-location', 'B-person', 'I-person', 'B-product', 'I-product', 
                           'PAD']
        tag2idx = {t: i for i, t in enumerate(tag_values)}

        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        sentences = [" ".join(tokens) for tokens in dataset["tokens"]]
        encodings = tokenizer(sentences, truncation=True, padding=True)

        labels = []
        for i, label_seq in enumerate(dataset["ner_tags"]):
            label_seq = [tag2idx[tag_values[label]] for label in label_seq]
            padding_length = len(encodings['input_ids'][i]) - len(label_seq)
            labels.append(label_seq + ([tag2idx['PAD']] * padding_length))

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset


class OntoNotesDataModule(NERDataModuleBase):
    """
    LightningDataModule Class for the OntoNotes 5.0 dataset.
    """
    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the loaded dataset.
        """
        # Train/val/test split
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["validation"]

        # Truncation
        if self.truncate is not None:
            self.train_dataset = self._truncate(self.train_dataset)
            self.val_dataset = self._truncate(self.val_dataset)
        else:
            pass

        # Tokenization
        print("Before Tokenization:", len(self.train_dataset))
        self.train_dataset = self._tokenize(self.train_dataset)
        print("After Tokenization:", len(self.train_dataset))
        self.val_dataset = self._tokenize(self.val_dataset)


    def _tokenize(self, dataset) -> torch.utils.data.TensorDataset:
        """
        Tokenizes the dataset.
        """
        tag_values = ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 
                      'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 
                      'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 
                      'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 
                      'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE', 'PAD']
        tag2idx = {t: i for i, t in enumerate(tag_values)}

        # Extract sentences and corresponding labels from the nested structure
        all_sentences = []
        all_labels = []

        for example in dataset:
            for sentence in example['sentences']:
                words = sentence['words']
                ner_tags = sentence['named_entities']
                all_sentences.append(" ".join(words))
                all_labels.append(ner_tags)

        # Truncate after flattening
        if self.truncate is not None:
            all_sentences = all_sentences[:self.truncate]
            all_labels = all_labels[:self.truncate]

        # Tokenizing the sentences
        tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        encodings = tokenizer(all_sentences, truncation=True, padding=True)

        labels = []
        for i, label_seq in enumerate(all_labels):
            padding_length = len(encodings['input_ids'][i]) - len(label_seq)
            labels.append(label_seq + ([tag2idx['PAD']] * padding_length))

        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

        return dataset