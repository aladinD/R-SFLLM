import pytest
import torch
from src.aggregator import Aggregator
from src.models.bert_module import CustomBertModelModule
from torch.nn.parameter import Parameter
from pytorch_lightning.utilities.parsing import AttributeDict

@pytest.fixture
def aggregator():
    """
    Fixture for creating an Aggregator instance.
    """
    # Initialize the Aggregator instance for testing
    return Aggregator(name="test_aggregator")


def test_accumulate_attentions(aggregator):
    # Create a list of dummy client models for testing
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })

    client_models = [CustomBertModelModule.from_pretrained(**model_config) for _ in range(3)]

    # Call the accumulate_attentions method and get the results
    attentions = aggregator.accumulate_attentions(client_models)

    # Check if the attentions list contains the expected number of dictionaries
    assert len(attentions) == len(client_models)

    # Check if each dictionary in the attentions list contains only "bert.encoder" parameters
    for attention in attentions:
        for param_name in attention.keys():
            assert param_name.startswith("bert.encoder")


def test_accumulate_heads(aggregator):
    # Create a list of dummy client models for testing
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })

    client_models = [CustomBertModelModule.from_pretrained(**model_config) for _ in range(3)]

    # Call the accumulate_heads method and get the results
    heads = aggregator.accumulate_heads(client_models)

    # Check if the heads list contains the expected number of dictionaries
    assert len(heads) == len(client_models)

    # Check if each dictionary in the heads list contains only "classifier" and "bert.pooler" parameters
    for head in heads:
        for param_name in head.keys():
            assert param_name.startswith("classifier") or param_name.startswith("bert.pooler")


def test_accumulate_embeddings(aggregator):
    # Create a list of dummy client models for testing
    model_config = AttributeDict({
        "pretrained_model_name_or_path": "bert-base-uncased",
        "num_labels": 2
    })

    client_models = [CustomBertModelModule.from_pretrained(**model_config) for _ in range(3)]

    # Call the accumulate_embeddings method and get the results
    embeddings = aggregator.accumulate_embeddings(client_models)

    # Check if the embeddings list contains the expected number of dictionaries
    assert len(embeddings) == len(client_models)

    # Check if each dictionary in the embeddings list contains only "bert.embeddings" parameters
    for embedding in embeddings:
        for param_name in embedding.keys():
            assert param_name.startswith("bert.embeddings")


def test_aggregate(aggregator):
    # Create a list of dummy gradients for testing
    client_gradients = [
        {
            "param1": Parameter(torch.randn(10, 5)),
            "param2": Parameter(torch.randn(5, 3)),
        },
        {
            "param1": Parameter(torch.randn(10, 5)),
            "param2": Parameter(torch.randn(5, 3)),
        },
    ]

    # Call the aggregate method and get the aggregated gradients
    aggregated_gradients = aggregator.aggregate(client_gradients)

    # Check if the aggregated_gradients dictionary contains the expected keys
    assert set(aggregated_gradients.keys()) == set(client_gradients[0].keys())

    # Check if the aggregated_gradients values are correct (averaged gradients)
    num_clients = len(client_gradients)
    for param_name in aggregated_gradients:
        expected_average = torch.zeros_like(client_gradients[0][param_name].data)
        for grad_dict in client_gradients:
            expected_average += grad_dict[param_name].data
        expected_average /= num_clients
        assert torch.allclose(aggregated_gradients[param_name], expected_average)
