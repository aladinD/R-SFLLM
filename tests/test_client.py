import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader
from src.models.bert_module import CustomBertModelModule
from src.client import Client

@pytest.fixture
def client():
    """
    Fixture for creating a Client instance.
    """
    # Create a sample model and trainer for testing
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })

    model = CustomBertModelModule.from_pretrained(**model_config)
    trainer = pl.Trainer(fast_dev_run=True)

    # Create dummy data for training and validation
    train_data = DataLoader(torch.randn(100, 3), batch_size=32)
    val_data = DataLoader(torch.randn(20, 3), batch_size=10)

    # Initialize the Client instance for testing
    return Client(id=1, model=model, trainer=trainer, train_data=train_data, val_data=val_data)
        

def test_update_model(client):
    # Create dummy weight updates as a dictionary
    updates = {
        "layer1.weight": torch.randn(10, 20),
        "layer2.weight": torch.randn(5, 10),
    }

    # Update the model with the dummy weight updates
    client.update_model(updates)

    # Check if the model parameters have been updated correctly
    for gradient_name, updated_param in updates.items():
        for name, param in client.model.named_parameters():
            if name == gradient_name:
                assert torch.allclose(param.data, updated_param)
