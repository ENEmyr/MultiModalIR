import os, pathlib, torchaudio, torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict


class MiniSpeechCommands(Dataset):
    def __init__(self, target_dir: str, transform) -> None:
        self.paths = list(pathlib.Path(target_dir).glob("*/*.wav"))
        self.target_dir = target_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.__find_class()

    def __find_class(self) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted([entry.name for entry in list(os.scandir(self.target_dir))])
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        return classes, class_to_idx

    def __load_audio(self, index: int) -> torch.Tensor:
        # metadata = torchaudio.info(self.paths[index])
        wav, _ = torchaudio.load(self.paths[index])
        return wav

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        wav = self.__load_audio(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        spec = self.transform(wav).squeeze(0).transpose(0, 1)
        return spec, class_idx
