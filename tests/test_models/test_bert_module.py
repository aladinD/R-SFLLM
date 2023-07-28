import pytest
import torch
from src.models.bert_module import CustomBertModelModule
from src.datamodules.bert_datamodule import BertDataModule
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning import Trainer


@pytest.fixture
def custom_bert_model():
    """
    Fixture for creating an instance of CustomBertModelModule.
    """
    config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })
    return CustomBertModelModule.from_pretrained(**config)


def test_forward(custom_bert_model):
    # Create some dummy input data for testing
    input_ids = torch.randint(1000, (32, 128))  # Example: Batch size 32, sequence length 128
    attention_mask = torch.ones(32, 128)        # All ones for complete attention
    labels = torch.randint(2, (32,))            # Example: Binary classification labels

    # Call the forward method and check if the output is as expected
    output = custom_bert_model(input_ids, attention_mask=attention_mask, labels=labels)
    assert "loss" in output    # The output should contain a 'loss' key
    assert "logits" in output  # The output should contain a 'logits' key


def test_training_step(custom_bert_model):
    # Create some dummy input data for testing
    batch = (torch.randint(1000, (32, 128)), torch.ones(32, 128), torch.randint(2, (32,)))

    # Call the training_step method and check if the loss is returned as expected
    loss = custom_bert_model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)  # The loss should be a torch Tensor


def test_validation_step(custom_bert_model):
    # Create some dummy input data for testing
    batch = (torch.randint(1000, (32, 128)), torch.ones(32, 128), torch.randint(2, (32,)))

    # Call the validation_step method and check if the loss is returned as expected
    loss = custom_bert_model.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)  # The loss should be a torch Tensor


def test_configure_optimizers(custom_bert_model):
    # Call the configure_optimizers method and check if an optimizer is returned
    optimizer = custom_bert_model.configure_optimizers()
    assert optimizer is not None  # The optimizer should not be None


def test_training_with_trainer():
    # CustomBertModelModule instance
    model_config = AttributeDict({
    "pretrained_model_name_or_path": "bert-base-uncased",
    "num_labels": 2
    })
    model = CustomBertModelModule.from_pretrained(**model_config)

    # Small datamodule instance
    data_config = AttributeDict({
        "glue_dataset": "sst2",
        "batch_size": 32,
        "num_workers": 1,
        "model_type": "bert-base-uncased",
        "truncate": 10
    })
    datamodule = BertDataModule(**data_config)

    # Create a trainer instance with some basic configurations
    trainer = Trainer(fast_dev_run=True)

    # Fit the model using the trainer and data module
    trainer.fit(model, datamodule=datamodule)