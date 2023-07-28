import pytest
import torch
from transformers import BertTokenizer
from src.datamodules.bert_datamodule import BertDataModule
from pytorch_lightning.utilities.parsing import AttributeDict
from datasets import load_dataset, DatasetDict
import numpy as np
# from tests.helpers import DummyDataset, DummyDataModule  # Create a dummy data module for testing


@pytest.fixture
def bert_data_module():
    """
    Fixture for creating an instance of BertDataModule.
    """
    config = AttributeDict({
        "glue_dataset": "sst2",
        "batch_size": 32,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "truncate": 10
    })
    return BertDataModule(**config)


class TestBertDataModule:
    """
    Test class for the BertDataModule.
    """
    def test_prepare_data(self, bert_data_module):
        # Call the prepare_data method
        bert_data_module.prepare_data()
        assert bert_data_module.dataset is not None  # The dataset should be loaded


    def test_truncation(self, bert_data_module):
        # Create a dummy dataset for testing
        dataset = load_dataset('glue', "sst2")
        bert_data_module.dataset = dataset

        # Assign the example of the train dataset
        bert_data_module.train_dataset = bert_data_module.dataset["train"]

        # Call the _truncate method and check if the truncation works as expected
        truncated_train_dataset = bert_data_module._truncate(bert_data_module.train_dataset)
        assert len(truncated_train_dataset) == bert_data_module.truncate

    
    def test_tokenization(self, bert_data_module):
        # Create a dummy dataset
        dataset = load_dataset('glue', "sst2")
        bert_data_module.dataset = dataset

        # Assign train and val datasets
        bert_data_module.train_dataset = bert_data_module.dataset["train"]
        bert_data_module.val_dataset = bert_data_module.dataset["validation"]

        # Call the _tokenize method and check if the tokenization works as expected
        for dataset in [bert_data_module.train_dataset, bert_data_module.val_dataset]:
            tokenized_dataset = bert_data_module._tokenize(dataset)
            assert isinstance(tokenized_dataset, torch.utils.data.TensorDataset)
            assert len(tokenized_dataset) == len(dataset)
            assert all(hasattr(data, "__getitem__") for data in tokenized_dataset)


    def test_setup(self, bert_data_module):
        # Create a dummy dataset for testing
        dataset = load_dataset('glue', "sst2")

        # Assign the dummy dataset to bert_data_module.dataset for testing
        bert_data_module.dataset = dataset

        # Call the setup method and check if the train_dataset and val_dataset are set
        bert_data_module.setup()

        assert bert_data_module.train_dataset is not None  # The train_dataset should be set
        assert bert_data_module.val_dataset is not None  # The val_dataset should be set


    def test_split_data(self, bert_data_module):
        # Create a dummy dataset for testing
        dataset = load_dataset('glue', "sst2")
        bert_data_module.dataset = dataset

        # Assign the train dataset
        bert_data_module.train_dataset = bert_data_module.dataset["train"]

        print("TRAIN : ", len(bert_data_module.train_dataset))

        # Call the _split_data method with num_splits=2 and check if the dataset is split
        num_splits = 3
        splits = bert_data_module._split_data(bert_data_module.train_dataset, num_splits=num_splits)

        assert isinstance(splits, list)
        assert len(splits) == num_splits
        assert all(isinstance(split, torch.utils.data.Subset) for split in splits)
        assert sum(len(split) for split in splits) == len(bert_data_module.train_dataset)

    
    def test_train_dataloader(self, bert_data_module):
        # Call the train_dataloader method and check if a DataLoader object is returned
        bert_data_module.prepare_data()
        bert_data_module.setup()
        dataloader = bert_data_module.train_dataloader()
        assert isinstance(dataloader, torch.utils.data.DataLoader)


    def test_val_dataloader(self, bert_data_module):
        # Call the val_dataloader method and check if a DataLoader object is returned
        bert_data_module.prepare_data()
        bert_data_module.setup()
        dataloader = bert_data_module.val_dataloader()
        assert isinstance(dataloader, torch.utils.data.DataLoader)
