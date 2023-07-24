import warnings
from typing import Any, Dict, List, Literal

import torch
import torch.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression.mae import MeanAbsoluteError

from .components.simple_atoa_net import SimpleATOANet


class ATOAModule(LightningModule):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        in_shape,
        embed_dim,
        num_paths,
        w_aoa: float = 1.0,
        w_toa: float = 1.0,
    ) -> None:

        super().__init__()
        if net is None:
            self.net = SimpleATOANet(in_shape=in_shape, embed_dim=embed_dim, num_paths=num_paths)
        else:
            self.net = net
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # l1 criterion saved for use in actual loss function
        self.l1_criterion = torch.nn.L1Loss(reduction="mean")

        # objects which aggregate the loss from each batch.
        # these are needed in cases when we train on multiple devices
        # since care must be taken when aggregating the losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # task metrics, in this case MAE for the path delays
        self.train_metric = MeanAbsoluteError()
        self.val_metric = MeanAbsoluteError()
        self.test_metric = MeanAbsoluteError()

        # for tracking best validation mae so far
        self.val_metric_best = MinMetric()

    def loss_fn(self, target: Dict[str, Any], preds: Dict[str, Any]) -> torch.Tensor:
        """Compute a weighted loss between the l1-norm for the path delays and the l1-norm for the
        aoas.

        :param target: Ground truth as dictionary with "path_delays" and "aoas" as keys, and the
        batch predictions for these values.
        :type target: Dict[str, Any]
        :param preds: Predictions
        :type preds: Dict[str, Any]
        :return: loss for each value in the batch
        :rtype: torch.Tensor
        """
        toas_target, toas_pred = target["path_delays"], preds["path_delays"]
        toa_loss = self.hparams.w_toa * self.l1_criterion(toas_pred, toas_target)

        aoas_target, aoas_pred = target["aoas"], preds["aoas"]
        aoa_loss = self.hparams.w_aoa * self.l1_criterion(aoas_pred, aoas_target)

        loss = toa_loss + aoa_loss
        return loss

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)

    def model_step(self, batch: Any):
        """Computes the predictions for a batch and the loss function batchwise."""
        x, targets = batch
        preds = self.forward(x)
        loss = self.loss_fn(target=targets, preds=preds)
        return loss, preds, targets

    def stage_step(self, batch: Any, stage: Literal["train", "val", "test"]):
        """Computes the predictions, averages losses over a batch, and computes the task metrics
        for training, validation, and testing stages."""
        stage_loss = getattr(self, f"{stage}_loss")
        stage_metric = getattr(self, f"{stage}_metric")

        loss, preds, targets = self.model_step(batch)
        stage_loss(loss)
        stage_metric(preds["path_delays"], targets["path_delays"])
        self.log(f"{stage}/loss", stage_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/metric", stage_metric, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        return self.stage_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.stage_step(batch, "val")

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_metric.compute()  # get current val acc
        self.val_metric_best(mae)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/metric_best", self.val_metric_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        return self.stage_step(batch, "test")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
