from typing import Any, Dict, Tuple

from torchvision import transforms, datasets
from torch import Tensor
from torch.utils.data import DataLoader
from rich.console import Console
from src.dataloaders import CustomDataloader

console = Console()


class MiniHandsignCommandsDataloader(CustomDataloader):

    transform: dict | Any

    def __init__(
        self, config: dict, dataset_split_paths: Dict[str, str], verbose: bool = True
    ) -> None:
        super().__init__(dataset_split_paths, config, verbose)
        self.transform = {
            k.upper(): transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            )
            for k in dataset_split_paths.keys()
        }
        self.handsign_datasets = {
            k.upper(): datasets.ImageFolder(v, transform=self.transform[k.upper()])
            for k, v in self.dataset_split_paths.items()
        }
        self.dataloaders = {
            k.upper(): DataLoader(
                v,
                batch_size=config["batch_size"],
                shuffle=(k == "TRAIN"),
                num_workers=config["num_workers"],
            )
            for k, v in self.handsign_datasets.items()
        }

        self.dataset_sizes = {k: len(v) for k, v in self.handsign_datasets.items()}
        self.class_names = self.handsign_datasets["TRAIN"].classes
        self._verbose()

    @property
    def train(self) -> DataLoader:
        return self.dataloaders["TRAIN"]

    @property
    def test(self) -> DataLoader:
        return self.dataloaders["TEST"]

    @property
    def validate(self) -> DataLoader:
        key = "VAL" if "VAL" in self.dataloaders.keys() else "VALIDATE"
        return self.dataloaders[key]

    def _collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        return Tensor(), Tensor()

    def _verbose(self) -> None:
        for k, v in self.dataset_sizes.items():
            console.log("Loaded {} images under {}".format(v, k))
        console.log("Classes: ", self.class_names)
