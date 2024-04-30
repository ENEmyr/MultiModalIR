import torch
import platform
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch import loggers as pl_loggers
from torch import optim, nn
from torcheval.metrics import MulticlassAccuracy
from src.models.VGG16 import VGG16


class VGG16Trainer(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VGG16(**config)
        if platform.system() == "Linux":
            # torch.compile requires Triton but currently Triton only supported Linux
            self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy()
        self.config = config

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        outputs = self.model(X)
        _, y_preds = outputs.max(1)
        loss = self.criterion(y_preds, y)

        self.accuracy.update(y_preds, y)
        acc = self.accuracy.compute()
        # acc = torch.sum(y_preds == y.data).item() / (len(y) * 1.0)

        self.log_dict(
            {"train_loss": loss, "train_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.accuracy.reset()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        outputs = self.model(X)
        _, y_preds = outputs.max(1)
        loss = self.criterion(y_preds, y)

        self.accuracy.update(y_preds, y)
        acc = self.accuracy.compute()

        self.log_dict(
            {"val_loss": loss, "val_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_end(self) -> None:
        self.accuracy.reset()

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        outputs = self.model(X)
        _, y_preds = outputs.max(1)
        loss = self.criterion(y_preds, y)

        self.accuracy.update(y_preds, y)
        acc = self.accuracy.compute()

        self.log_tb_images((X, y, y_preds, batch_idx))
        self.log_dict(
            {"test_loss": loss, "test_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_test_end(self) -> None:
        self.accuracy.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.config["optimizer"].upper() == "ADAM":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                betas=self.config["betas"],
                eps=self.config["eps"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
            )
        return optimizer

    def log_tb_images(self, viz_batch) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
            tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
            tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)
