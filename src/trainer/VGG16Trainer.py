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
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        y_pred_argmax = torch.argmax(y_pred, dim=1).to(torch.float32)
        y_argmax = torch.argmax(y, dim=1).to(torch.float32)

        self.accuracy.update(y_pred_argmax, y_argmax)
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
        y_pred = self.model(X)
        y_pred_argmax = torch.argmax(y_pred, dim=1).to(torch.float32)
        y_argmax = torch.argmax(y, dim=1).to(torch.float32)
        loss = self.criterion(y_pred, y)

        self.accuracy.update(y_pred_argmax, y_argmax)
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
        y_pred = self.model(X)
        y_pred_argmax = torch.argmax(y_pred, dim=1).to(torch.float32)
        y_argmax = torch.argmax(y, dim=1).to(torch.float32)
        loss = self.criterion(y_pred, y)

        self.accuracy.update(y_pred_argmax, y_argmax)
        acc = self.accuracy.compute()

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
