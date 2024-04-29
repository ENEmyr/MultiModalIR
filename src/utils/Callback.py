import lightning as L
from lightning.pytorch.callbacks import Callback
from rich import console

console = console.Console()


class VerboseCallback(Callback):
    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule"):
        console.log("\n= Start training =\n")

    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule"):
        console.log("\n= Stop training =\n")

    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        console.log("\n= Start validating =\n")

    def on_validation_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        console.log("\n= Stop validating =\n")

    def on_test_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        console.log("\n= Start testing =\n")

    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        console.log("\n= Stop testing =\n")
