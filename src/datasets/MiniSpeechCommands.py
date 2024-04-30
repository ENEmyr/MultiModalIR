import os
import pathlib
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from typing import Tuple, List, Dict
from src.utils.TextTransform import TextTransform


class MiniSpeechCommands(Dataset):
    def __init__(
        self, target_dir: str, transform=None, text_transform: bool = False
    ) -> None:
        self.paths = list(pathlib.Path(target_dir).glob("*/*.wav"))
        self.target_dir = target_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.__find_class()
        self.text_transform = text_transform

    def __find_class(self) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted([entry.name for entry in list(os.scandir(self.target_dir))])
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        return classes, class_to_idx

    def __load_audio(self, index: int) -> Tuple[torch.Tensor, int]:
        # metadata = torchaudio.info(self.paths[index])
        wav, sr = torchaudio.load(self.paths[index])
        return wav, sr

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor | int]:
        wav, sr = self.__load_audio(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        label = None
        if self.text_transform:
            tt = TextTransform()
            label = torch.tensor(tt.text_to_int(class_name.upper()))
        else:
            label = one_hot(
                torch.tensor([class_idx]), num_classes=len(self.class_to_idx)
            )
        if self.transform is not None:
            if (
                str(self.transform.__class__)
                == "<class 'transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2Processor'>"
            ):
                wav = self.transform(wav, sampling_rate=sr, return_tensors="pt")
                assert len(wav) == 1  # assure to only return input_values
                wav = wav.input_values
                wav = wav.squeeze(0)
            else:
                wav = self.transform(wav).squeeze(0).transpose(0, 1)
            return wav, label
        wav = self.__pad_seq(wav).squeeze(0)
        return wav, label

    def __pad_seq(self, seq: torch.Tensor) -> torch.Tensor:
        num_frames = seq.shape[1]
        if num_frames > 16000:
            seq = seq[:, :16000]
        else:
            pad_width = 16000 - num_frames
            seq = np.pad(seq, [(0, 0), (0, pad_width)], mode="constant")
        return seq

    def decode_label(self, pred: int | torch.Tensor) -> str:
        if isinstance(pred, torch.Tensor):
            pred = int(pred.argmax(pred.ndim - 1).item())
        return self.classes[pred]
