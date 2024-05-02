import torch
import platform
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import optim, nn
from torcheval.metrics import MulticlassAccuracy
import lightning as L
from src.models.Wav2Vec2ConformerCls import Wav2Vec2ConformerCls


# TODO: maybe need to explicit clarify what inside config for the sake of save_hyperparameters()
class Wav2Vec2ConformerTrainer(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = Wav2Vec2ConformerCls(**config)
        if platform.system() == "Linux":
            # torch.compile requires Triton but currently Triton only supported Linux
            self.model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy()
        self.config = config

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, loss, acc = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {"train_loss": loss, "train_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_preds_argmax, loss, acc = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {"val_loss": loss, "val_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return y_preds_argmax

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_preds_argmax, loss, acc = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {"test_loss": loss, "test_accuracy": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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
        # acc = torch.sum(y_pred == y).item() / (len(y) * 1.0)
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

    def on_train_epoch_end(self) -> None:
        self.accuracy.reset()

    def on_validation_end(self) -> None:
        self.accuracy.reset()

    def on_test_end(self) -> None:
        self.accuracy.reset()
