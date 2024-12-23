import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from typing import Any, Dict, Tuple
from rich.console import Console
from src.dataloaders import CustomDataloader
from src.datasets.Flickr8KWithAudioVector import Flickr8KWithAudioVector

console = Console()


class Flickr8KWithAudioVectorDataloader(CustomDataloader):

    transform: dict | Any

    def __init__(
        self,
        dataset_split_ratio: Dict[str, float],
        dataset_params: Dict[str, str],
        batch_size: int,
        verbose: bool = True,
    ) -> None:
        super().__init__(dataset_split_ratio, dataset_params, verbose)

        dataset = Flickr8KWithAudioVector(**dataset_params)

        # Split dataset into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(total_size * dataset_split_ratio["train"])
        val_size = int(total_size * dataset_split_ratio["val"])
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create dataloaders
        self.dataloaders = {
            "TRAIN": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            "VAL": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            "TEST": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        }
        if verbose:
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

    def _verbose(self) -> None:
        for k, v in self.dataloaders.items():
            console.log(
                "Loaded {} images/audios under {} set".format(len(v.dataset), k)
            )
        batch = next(iter(self.dataloaders["TRAIN"]))
        images, waveforms = batch["image"], batch["audio"]
        console.log(
            f"Train Batch - Images shape: {images['embed'].shape}, Waveforms shape: {waveforms['embed'].shape}",
            f"\nTrain Batch - Images attention_mask shape: {images['attention_mask'].shape}, Waveforms attention_mask shape: {waveforms['attention_mask'].shape}",
        )

    def _collate_fn(self, batch) -> Tuple:
        return ()
