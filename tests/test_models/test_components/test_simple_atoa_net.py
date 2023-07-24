import pytest
import torch

from src.models.components.simple_atoa_net import SimpleATOANet


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("in_shape", [(32, 64, 14), (16, 16, 16)])
@pytest.mark.parametrize("embed_dim", [None, 16])
@pytest.mark.parametrize("num_paths", [3, 5])
def test_simple_atoa_net(batch_size, in_shape, embed_dim, num_paths):
    x = torch.randn([batch_size, *in_shape])
    net = SimpleATOANet(
        in_shape=in_shape, embed_dim=embed_dim, num_paths=num_paths, initialize_lazy_layers=False
    )

    assert len([p for p in net.parameters() if isinstance(p, torch.nn.UninitializedParameter)]) > 0

    preds = net.forward(x)
    assert all(len(p) == batch_size for p in preds.values())
    assert (
        len([p for p in net.parameters() if isinstance(p, torch.nn.UninitializedParameter)]) == 0
    )

    net = SimpleATOANet(
        in_shape=in_shape, embed_dim=embed_dim, num_paths=num_paths, initialize_lazy_layers=True
    )
    assert (
        len([p for p in net.parameters() if isinstance(p, torch.nn.UninitializedParameter)]) == 0
    )
