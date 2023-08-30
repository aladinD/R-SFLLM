from typing import List, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils
from src.datamodules.simulation_datamodule import SimulationDatamodule

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def convert(cfg: DictConfig) -> Tuple[dict, dict]:
    """Converts model in given checkpoint to SNN and tests on a datamodule testset.
    It does the following:
    
    1. Load checkpoint path and extract nn.Module corresponding to network architecture.
    2. Extract dataloader used for SNN conversion
    3. Convert to ANN to SNN and save the weights of the converted SNN.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting conversion!")
    datamodule: SimulationDatamodule = datamodule.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path
    )
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    converter = utils.snn_convert.SNNConverter(mode="max", dataloader=train_dataloader)

    log.info("Loading model with ckpt parameters!")
    model = model.load_from_checkpoint(cfg.ckpt_path)
    net = model.net

    log.info("Starting conversion!")
    snn = converter(net)
    save_path = os.path.join(os.path.split(cfg.ckpt_path)[0], "converted_snn.pt")
    log.info(f"Done conversion. Saving converted model to: {save_path}")
    torch.save(snn, save_path)

    log.info("Starting testing model!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metric_dict_cnn = trainer.callback_metrics

    log.info("Starting testing converted model!")
    model_snn = model
    model_snn.net = snn
    trainer.test(model=model_snn, datamodule=datamodule)
    metric_dict_snn = trainer.callback_metrics

    metric_dict = {"cnn": metric_dict_cnn, "snn": metric_dict_snn}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig | None = None) -> None:
    convert(cfg)


if __name__ == "__main__":
    main()
