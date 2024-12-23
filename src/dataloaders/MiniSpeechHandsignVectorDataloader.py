from typing import Dict, Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from rich.console import Console
from src.datasets.MiniSpeechHandsignVector import MiniSpeechHandsignVector
from src.dataloaders import CustomDataloader

console = Console()


class MiniSpeechHandsignVectorDataloader(CustomDataloader):
    speech_datasets: dict

    def __init__(
        self,
        config: dict,
        dataset_split_paths: Dict[str, str],
        padding: str = "zero",
        verbose: bool = True,
    ) -> None:
        super().__init__(dataset_split_paths, config, verbose)
        self.speech_datasets = {
            k.upper(): MiniSpeechHandsignVector(
                v,
                padding=padding,
                text_transform=False,
            )
            for k, v in self.dataset_split_paths.items()
        }

        self.dataloaders = {
            k: torch.utils.data.DataLoader(
                v,
                batch_size=config["dataloader_params"]["batch_size"],
                shuffle=(k != "VAL"),
                num_workers=config["num_workers"],
                collate_fn=self._collate_fn,
            )  # os.cpu_count() = 24
            for k, v in self.speech_datasets.items()
        }
        self.dataset_sizes = {k: len(v) for k, v in self.speech_datasets.items()}
        self.class_names = list(self.speech_datasets.values())[0].classes
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

    def _collate_fn(self, batch) -> Dict[str, Dict[str, torch.Tensor]]:
        images = []
        audios = []
        image_mask = []
        audio_mask = []
        labels = []
        ids = []
        for vector in batch:
            images.append(vector["image"]["embed"])
            audios.append(vector["audio"]["embed"])
            image_mask.append(vector["image"]["attention_mask"])
            audio_mask.append(vector["audio"]["attention_mask"])
            labels.append(vector["label"])
            ids.append(vector["id"])
        vector = {
            "image": {
                "embed": torch.stack(images),
                "attention_mask": torch.stack(image_mask),
            },
            "audio": {
                "embed": torch.stack(audios),
                "attention_mask": torch.stack(audio_mask),
            },
        }
        vector["label"] = (torch.stack(labels)).squeeze().to(torch.float32)
        vector["id"] = torch.Tensor(ids).to(torch.float32)
        return vector

    def _verbose(self) -> None:
        for k, v in self.dataset_sizes.items():
            console.log("Loaded {} image/audio vectors under {}".format(v, k))
        console.log("Classes: ", self.class_names)
