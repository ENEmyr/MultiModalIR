import pytest
import json
from torch import randn
from os.path import join

from torch.utils.data import DataLoader
from src.dataloaders.MiniHandsignCommandsDataloader import (
    MiniHandsignCommandsDataloader,
)


@pytest.fixture
def config():
    with open("./configs/VGG16.json", "r") as f:
        cfg = json.load(f)
    return cfg


@pytest.fixture
def dataset_split_paths(config):
    return {
        "train": join(config["dataset_path"], "train"),
        "test": join(config["dataset_path"], "test"),
        "val": join(config["dataset_path"], "val"),
    }


@pytest.fixture
def dataloaders(config, dataset_split_paths):
    dtl = MiniHandsignCommandsDataloader(
        config=config, dataset_split_paths=dataset_split_paths, verbose=True
    )
    return dtl


def test_dataloader_isinstance(dataloaders):
    assert isinstance(dataloaders.train, DataLoader)


def test_batch_shape(config, dataloaders):
    inputs, labels = next(iter(dataloaders.train))
    assert inputs.shape == randn(config["batch_size"], 3, 224, 224).shape
    assert labels.shape == randn(config["batch_size"], 8).shape
