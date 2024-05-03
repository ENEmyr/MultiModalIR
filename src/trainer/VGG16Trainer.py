import os
import platform
import random
from typing import Any

import lightning as L
import torch
import sklearn.metrics
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim
from torcheval.metrics import MulticlassAccuracy

from src.models.VGG16 import VGG16
from src.utils.Plot import image_grid, buf_figure_to_image, plot_confusion_matrix


class VGG16Trainer(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = VGG16(**config)
        if platform.system() == "Linux":
            # torch.compile requires Triton but currently Triton only supported Linux
            self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy()
        self.config = config
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, loss, acc = self.__get_preds_loss_accuracy(batch)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": acc},
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"],
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_preds_argmax, loss, acc = self.__get_preds_loss_accuracy(batch)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": acc},
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"],
        )
        return y_preds_argmax

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_preds_argmax, loss, acc = self.__get_preds_loss_accuracy(batch)
        self.log_dict(
            {"test_loss": loss, "test_accuracy": acc},
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"],
        )
        return y_preds_argmax

    def __get_preds_loss_accuracy(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        y_pred_argmax = torch.argmax(y_pred, dim=1).to(torch.float32)
        y_argmax = torch.argmax(y, dim=1).to(torch.float32)

        self.accuracy.update(y_pred_argmax, y_argmax)
        acc = self.accuracy.compute()
        # acc = torch.sum(y_preds == y.data).item() / (len(y) * 1.0)
        return y_pred_argmax, loss, acc

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

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_end(self) -> None:
        self.accuracy.reset()

    def on_validation_end(self) -> None:
        self.accuracy.reset()

    def on_test_end(self) -> None:
        self.accuracy.reset()

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        X, y = batch
        y_argmax = torch.argmax(y, dim=1).to(torch.int)
        y_preds_argmax = outputs.to(torch.int)
        cm = sklearn.metrics.confusion_matrix(y_argmax.cpu(), y_preds_argmax.cpu())
        labels_decode = sorted(
            [
                entry.name
                for entry in list(
                    os.scandir(os.path.join(self.config["dataset_path"], "test"))
                )
            ]
        )
        sample_idxs = random.sample(range(len(X)), 25)
        titles = [
            f"G: {labels_decode[y_argmax[i]]} - P: {labels_decode[y_preds_argmax[i]]}"
            for i in sample_idxs
        ]
        images = [
            (X[i].cpu().numpy() * 255).transpose(1, 2, 0).astype(int)
            for i in sample_idxs
        ]
        figure_grid = image_grid(titles, images)
        figure_cm = plot_confusion_matrix(cm, labels_decode)
        img_grid = torch.Tensor(buf_figure_to_image(figure_grid))
        img_cm = torch.Tensor(buf_figure_to_image(figure_cm))
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(
                    key="test_samples",
                    images=[img_grid],
                    caption=[f"Testing Samples at Batch {batch_idx}"],
                )
                logger.log_image(
                    key="confusion_matrix",
                    images=[img_cm],
                    caption=["Confusion Matrix"],
                )
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(
                    f"Testing Samples at Batch {batch_idx}",
                    figure_grid,
                    self.global_step,
                )
                logger.experiment.add_figure(
                    f"Confusion Matrix", figure_cm, self.global_step
                )
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
