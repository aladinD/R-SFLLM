import pytest
import torch

from src.datamodules.components.transforms.target_transforms import ParseAngles


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("azimuth_only", [True, False])
@pytest.mark.parametrize("elevation_only", [True, False])
def test_parse_angles(
    normalize: bool,
    azimuth_only: bool,
    elevation_only: bool,
    batch_size: int = 32,
    num_paths: int = 5,
):
    x = torch.randn(batch_size, 10)
    az = torch.pi - 2 * torch.pi * torch.rand(batch_size, num_paths)
    el = 0.5 * torch.pi - torch.pi * torch.rand(batch_size, num_paths)
    angs = torch.stack([az, el], dim=-1)

    # we use the same angles since we do not care about the actual values
    y = {"aoas": angs, "aods": angs}

    # we make a copy since this transformation acts in place
    y_cpy = y.copy()

    # check whether wrong input raises error
    if (azimuth_only is True) and (elevation_only is True):
        with pytest.raises(AssertionError):
            parse_angle = ParseAngles(
                normalize=normalize, azimuth_only=azimuth_only, elevation_only=elevation_only
            )
    else:
        parse_angle = ParseAngles(
            normalize=normalize, azimuth_only=azimuth_only, elevation_only=elevation_only
        )
        x_new, y_new = parse_angle((x, y))

        # check whether input tensor remains the same
        assert torch.allclose(x, x_new)

        # check whether normalization is done properly
        if normalize:
            assert all(
                (torch.all(y_new[name]) <= 1.0) and (torch.all(y_new[name]) >= -1.0)
                for name in parse_angle.names
            )

        # check whether flags return correct shapes
        if azimuth_only != elevation_only:
            assert all(
                y_new[name].shape == torch.Size([batch_size, num_paths])
                for name in parse_angle.names
            )
        else:
            assert all(y_new[name].shape == y_cpy[name].shape for name in parse_angle.names)
