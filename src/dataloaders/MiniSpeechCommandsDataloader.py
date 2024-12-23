from typing import Any, Dict, Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PreTrainedTokenizerFast
from rich.console import Console
from src.datasets.MiniSpeechCommands import MiniSpeechCommands
from src.dataloaders import CustomDataloader

console = Console()


class MiniSpeechCommandsDataloader(CustomDataloader):
    transform: PreTrainedTokenizerFast | Any
    speech_datasets: dict

    def __init__(
        self, config: dict, dataset_split_paths: Dict[str, str], verbose: bool = True
    ) -> None:
        super().__init__(dataset_split_paths, config, verbose)
        self.transform = AutoProcessor.from_pretrained(
            "facebook/wav2vec2-conformer-rope-large-960h-ft",
            return_attention_mask=False,
        )
        self.speech_datasets = {
            k.upper(): MiniSpeechCommands(
                v,
                transform=self.transform,
                text_transform=False,
            )
            for k, v in self.dataset_split_paths.items()
        }

        self.dataloaders = {
            k: torch.utils.data.DataLoader(
                v,
                batch_size=config["batch_size"],
                shuffle=(k != "VAL"),
                num_workers=config["num_workers"],
                collate_fn=self._collate_fn,
            )  # os.cpu_count() = 24
            for k, v in self.speech_datasets.items()
        }
        self.dataset_sizes = {k: len(v) for k, v in self.speech_datasets.items()}
        self.class_names = list(self.speech_datasets.values())[0].classes
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

    def _collate_fn(self, batch) -> Tuple[Tensor, Tensor, list]:
        inputs = []
        labels = []
        file_paths = []
        for input, label, file_path in batch:
            inputs.append(input.T)
            labels.append(label)
            file_paths.append(file_path)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels = torch.stack(labels)
        return inputs.squeeze(), labels.squeeze().to(torch.float32), file_paths

    def _verbose(self) -> None:
        for k, v in self.dataset_sizes.items():
            console.log("Loaded {} audios under {}".format(v, k))
        console.log("Classes: ", self.class_names)
