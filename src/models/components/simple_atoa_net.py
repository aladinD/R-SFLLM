import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float)


class SimpleATOANet(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        embed_dim: Optional[int],
        num_params: int = 2,
        num_paths: int = 3,
        initialize_lazy_layers: bool = True,
    ) -> None:
        """Simple convnet to estimate delay and elevation AoA.

        :param in_shape: Input shape corresponding to (num_antennas, num_fft, num_symbols).
        :type in_shape: Tuple[int, int, int]
        :param embed_dim: Size of the FC output layer after conv block.
        :type embed_dim: Optional[int]
        :param num_params: Number of path parameters to estimate, defaults to 2
        :type num_params: int, optional
        :param num_paths: Number of paths for which parameters are estimated, defaults to 3
        :type num_paths: int, optional
        :param initialize_lazy_layers: Whether to initialize the conv layers lazily, defaults to True
        :type initialize_lazy_layers: bool, optional
        """
        assert num_params == 2, "Only two parameters are supported for now!"
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[0], out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # TODO: compute beforehand the dimensionality of features after convolutinoal layers!
        # TODO: get rid of lazy linear completely!
        # in_features = (in_shape[1] // 4) * (in_shape[2] // 4) * 128
        out_features = num_params * num_paths
        if embed_dim:
            self.head = nn.Sequential(
                nn.LazyLinear(out_features=embed_dim),
                nn.ReLU(),
                nn.Linear(in_features=embed_dim, out_features=out_features),
            )
        else:
            self.head = nn.LazyLinear(out_features=out_features)

        self.num_params = num_params
        self.num_paths = num_paths

        if initialize_lazy_layers:
            batch = torch.ones([1] + list(in_shape), dtype=torch.float32)
            self.forward(batch)

    def forward(self, x: torch.Tensor):
        batch_size = x.size()[0]

        # pass through convnet
        res_backbone = self.backbone.forward(x)

        # flatten and project to parameter space
        flattened = res_backbone.view(batch_size, -1)
        out_params = self.head(flattened)

        # process path delays and aoas into a dict
        path_delays, aoas = out_params[..., : self.num_paths], out_params[..., self.num_paths :]

        # path delays can not be negative and should be sorted!
        path_delays = torch.sort(F.relu(path_delays), dim=-1, descending=False)[0]
        # aoas should be in (-1, 1)
        aoas = torch.tanh(aoas)

        return {"path_delays": path_delays, "aoas": aoas}
