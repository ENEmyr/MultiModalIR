import os, pathlib, torchaudio, torch, librosa
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from typing import Tuple, List, Dict


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
