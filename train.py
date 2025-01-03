import argparse
import json
import os
import yaml
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from rich import console, pretty, traceback

from src.utils.Callback import VerboseCallback

console = console.Console()
pretty.install()
traceback.install(show_locals=False)
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


def MultiModalFusionTrainer(trainer: L.Trainer, config: dict):
    from src.trainer.MultiModalFusionTrainer import MultiModalFusionTrainer

    # from src.dataloaders.Flickr8KWithAudioVectorDataloader import (
    #     Flickr8KWithAudioVectorDataloader,
    # )

    # dataloaders = Flickr8KWithAudioVectorDataloader(**config["dataloader_params"])
    from src.dataloaders.MiniSpeechHandsignVectorDataloader import (
        MiniSpeechHandsignVectorDataloader,
    )

    dataset_split_paths = {
        x: os.path.join(
            config["dataloader_params"]["dataset_params"]["dataset_path"], x
        )
        for x in [TRAIN, TEST, VAL]
    }
    dataloaders = MiniSpeechHandsignVectorDataloader(
        config=config,
        dataset_split_paths=dataset_split_paths,
        padding=config["dataloader_params"]["dataset_params"]["padding"],
        verbose=True,
    )

    model = MultiModalFusionTrainer(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloaders.train,
        val_dataloaders=dataloaders.validate,
    )
    trainer.test(model=model, dataloaders=dataloaders.test, verbose=True)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.isfile(args.conf_path) and args.conf_path.split(".")[-1] in [
        "json",
        "yaml",
    ]
    if args.conf_path.split(".")[-1] == "yaml":
        with open(args.conf_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(args.conf_path, "r") as f:
            config = json.load(f)

    assert config["model"].upper() in (
        "Wav2Vec2ConformerCls".upper(),
        "Vgg16".upper(),
        "MultiModalFusion".upper(),
    )

    if "dataset_path" in config["dataloader_params"]["dataset_params"]:
        assert os.path.exists(
            config["dataloader_params"]["dataset_params"]["dataset_path"]
        )
    elif (
        "image_dir" in config["dataloader_params"]["dataset_params"]
        and "audio_dir" in config["dataloader_params"]["dataset_params"]
    ):
        assert os.path.exists(
            config["dataloader_params"]["dataset_params"]["image_dir"]
        ) and os.path.isdir(config["dataloader_params"]["dataset_params"]["audio_dir"])

    if not os.path.exists(config["weight_save_path"]):
        os.makedirs(config["weight_save_path"])

    loggers = []
    tb_logger = TensorBoardLogger("./logs/", name=config["model"])
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project="MultiModalFusion",
            name=config["model"] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M"),
            save_dir="./logs/",
            log_model=False,  # change to "all" to logged durring training, True to logged at the end of training
        )
        loggers.append(wandb_logger)
    loggers.append(tb_logger)

    if config["model"].upper() in ["MultiModalFusion".upper()]:
        ckpt_callback = ModelCheckpoint(
            monitor="val_mean_similarity",
            filename="{epoch}-{val_loss:.2f}-{val_mean_similarity:.2f}",
            auto_insert_metric_name=True,  # set this to False if metrics contain / otherise it will result in extra folders
            mode="max",
        )
    else:
        ckpt_callback = ModelCheckpoint(
            monitor="val_accuracy",
            filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
            auto_insert_metric_name=True,  # set this to False if metrics contain / otherise it will result in extra folders
            mode="max",
        )
    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["weight_save_path"],
        benchmark=config["benchmark"],
        precision=config["precision"],
        max_epochs=config["epochs"],
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=config["log_frequency"],
        logger=loggers,
        callbacks=[
            ckpt_callback,
            EarlyStopping(**config["early_stopping"]),
        ],
    )

    if config["model"].upper() == "Wav2Vec2ConformerCls".upper():
        Wav2VecConformerClsTrainer(trainer, config)
    elif config["model"].upper() == "VGG16":
        VGG16Trainer(trainer, config)
    elif config["model"].upper() == "MultiModalFusion".upper():
        MultiModalFusionTrainer(trainer, config)
    else:
        raise NotImplementedError(f"{config['model']} model not implemented")
