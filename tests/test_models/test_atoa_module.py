from functools import partial

import torch

from src.datamodules.components.datasets.simulation_dataset import SimulationDataset
from src.datamodules.components.transforms.core import ComplexToReal, Compose
from src.datamodules.components.transforms.target_transforms import ParseAngles
from src.datamodules.simulation_datamodule import SimulationDatamodule
from src.models.atoa_module import ATOAModule


def test_atoa_module(num_ant_rx=16, batch_size=32, num_symbols=14, num_carriers=256, num_paths=5):
    dset = SimulationDataset(
        num_ant_rx=num_ant_rx,
        num_symbols=num_symbols,
        num_carriers=num_carriers,
        num_paths=num_paths,
    )

    transforms = Compose([ComplexToReal(), ParseAngles(normalize=True, elevation_only=True)])
    dmod = SimulationDatamodule(dataset=dset, batch_size=batch_size, transforms=transforms)

    optim = partial(torch.optim.Adam, lr=1e-4)

    atoa_module = ATOAModule(
        net=None,
        optimizer=optim,
        in_shape=(2 * num_ant_rx, num_carriers, num_symbols),
        embed_dim=None,
        num_paths=num_paths,
        scheduler=None,
    )

    dmod.setup()
    tloader = dmod.train_dataloader()
    batch = next(iter(tloader))

    # check if loss returned is scalar, since it is averaged over batch size
    res: torch.Tensor = atoa_module.training_step(batch=batch, batch_idx=0)
    assert res["loss"].dim() == 0
