import librosa
import numpy as np
import torchaudio.transforms as T
import os, pathlib, torchaudio, torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn.functional import one_hot
from typing import Tuple, List, Dict, Union
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


class SpeechDataset(Dataset):
    def __init__(
        self,
        target_dir: str,
        freq_masking: int = -1,
        time_masking: int = -1,
        pad_length: int = 101,
    ) -> None:
        self.paths = list(pathlib.Path(target_dir).glob("*/*.wav"))
        self.target_dir = target_dir
        self.classes, self.class_to_idx = self.__find_class()
        self.pad_length = pad_length
        if freq_masking != -1 and time_masking != -1:
            self.transform = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=freq_masking),
                *[
                    T.TimeMasking(time_mask_param=time_masking, p=0.05)
                    for _ in range(10)
                ]
            )
        else:
            self.transform = None

    def __find_class(self) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted([entry.name for entry in list(os.scandir(self.target_dir))])
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]

        waveform, sample_rate = torchaudio.load(audio_path)
        # waveform = waveform[0]  # Mono-channel audio

        # Extract speech features (e.g., MFCCs) using Librosa
        mfcc = librosa.feature.mfcc(
            y=np.array(waveform),
            sr=sample_rate,
            n_mfcc=80,  # Number of MFCC coefficients
            hop_length=int(sample_rate * 0.01),  # Frame hop length (10 ms)
            n_fft=int(sample_rate * 0.025),  # Frame size (25 ms)
        )
        mfcc_tensor = torch.from_numpy(mfcc.copy()).float()

        # Apply frequency/time masking
        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)

        mfcc_tensor = self.pad_seq(mfcc_tensor).transpose(1, 0)

        return mfcc_tensor, class_idx

    def pad_seq(self, mfcc):
        num_frames = mfcc.shape[1]
        if num_frames > self.pad_length:
            mfcc = mfcc[:, : self.pad_length]
        else:
            pad_width = self.pad_length - num_frames
            mfcc = np.pad(mfcc, [(0, 0), (0, pad_width)], mode="constant")
        return mfcc
