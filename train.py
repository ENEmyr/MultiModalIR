import os
import argparse
import json
import lightning as L
from rich import console
from src.utils.Callback import VerboseCallback
from lightning.pytorch.loggers import TensorBoardLogger

console = console.Console()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    help="Path to config file",
    type=str,
    dest="conf_path",
    required=True,
)

parser.add_argument(
    "-w",
    "--use-wandb",
    help="Use Wandb as a tracker or not",
    action="store_true",
    dest="use_wandb",
    default=False,
)

TRAIN = "train"
VAL = "val"
TEST = "test"


def Wav2VecConformerClsTrainer(trainer: L.Trainer, config: dict):
    from src.trainer.Wav2Vec2ConformerTrainer import Wav2Vec2ConformerTrainer
    from src.dataloaders.MiniSpeechCommandsDataloader import (
        MiniSpeechCommandsDataloader,
    )

    dataset_split_paths = {
        x: os.path.join(config["dataset_path"], x) for x in [TRAIN, TEST, VAL]
    }
    dataloaders = MiniSpeechCommandsDataloader(
        config=config, dataset_split_paths=dataset_split_paths, verbose=True
    )

    model = Wav2Vec2ConformerTrainer(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloaders.train,
        val_dataloaders=dataloaders.validate,
    )
    trainer.test(model=model, dataloaders=dataloaders.test, verbose=True)


def VGG16Trainer(trainer: L.Trainer, config: dict):
    from src.trainer.VGG16Trainer import VGG16Trainer
    from src.dataloaders.MiniHandsignCommandsDataloader import (
        MiniHandsignCommandsDataloader,
    )

    dataset_split_paths = {
        x: os.path.join(config["dataset_path"], x) for x in [TRAIN, TEST, VAL]
    }
    dataloaders = MiniHandsignCommandsDataloader(
        config=config, dataset_split_paths=dataset_split_paths, verbose=True
    )

    model = VGG16Trainer(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloaders.train,
        val_dataloaders=dataloaders.validate,
    )
    trainer.test(model=model, dataloaders=dataloaders.test, verbose=True)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.isfile(args.conf_path) and args.conf_path.split(".")[-1] == "json"
    with open(args.conf_path, "r") as f:
        config = json.load(f)

    assert config["model"].upper() in (
        "Wav2Vec2ConformerCls".upper(),
        "Vgg16".upper(),
        "MultiFusion".upper(),
    )
    assert os.path.exists(config["dataset_path"])
    if not os.path.exists(config["weight_save_path"]):
        os.makedirs(config["weight_save_path"])

    tb_logger = TensorBoardLogger("./weights/", name=config["model"])
    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["weight_save_path"],
        benchmark=config["benchmark"],
        precision=config["precision"],
        max_epochs=config["epochs"],
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=config["log_frequency"],
        logger=tb_logger,
        # callbacks=VerboseCallback(),
    )

    if config["model"].upper() == "Wav2Vec2ConformerCls".upper():
        Wav2VecConformerClsTrainer(trainer, config)
    elif config["model"].upper() == "VGG16":
        VGG16Trainer(trainer, config)
    else:
        pass
