import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F


class Flickr8KWithAudio(Dataset):
    def __init__(
        self,
        image_dir,
        audio_dir,
        transform=None,
        audio_sample_rate=16000,
        max_audio_duration=1,
        **kwargs,
    ):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            audio_dir (str): Path to the directory containing audio files.
            transform (callable, optional): Optional transform to be applied on an image.
            audio_sample_rate (int): Target sample rate for audio files.
            max_audio_duration (int): Maximum audio duration in seconds.
        """
        self.image_dir = image_dir
        self.image_files = []
        self.audio_paths = {}
        for filename in sorted(os.listdir(image_dir)):
            rand_list = []
            filename_no_ext = filename.split(".")[0]
            if os.path.exists(os.path.join(audio_dir, f"{filename_no_ext}_0.wav")):
                rand_list.append(os.path.join(audio_dir, f"{filename_no_ext}_0.wav"))
            if os.path.exists(os.path.join(audio_dir, f"{filename_no_ext}_1.wav")):
                rand_list.append(os.path.join(audio_dir, f"{filename_no_ext}_1.wav"))
            if os.path.exists(os.path.join(audio_dir, f"{filename_no_ext}_2.wav")):
                rand_list.append(os.path.join(audio_dir, f"{filename_no_ext}_2.wav"))
            if os.path.exists(os.path.join(audio_dir, f"{filename_no_ext}_3.wav")):
                rand_list.append(os.path.join(audio_dir, f"{filename_no_ext}_3.wav"))
            if os.path.exists(os.path.join(audio_dir, f"{filename_no_ext}_4.wav")):
                rand_list.append(os.path.join(audio_dir, f"{filename_no_ext}_4.wav"))
            if len(rand_list) == 0:
                continue
            self.image_files.append(filename)
            self.audio_paths[filename] = random.choice(rand_list)

        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_samples = max_audio_duration * audio_sample_rate

        assert len(self.image_files) == len(
            self.audio_paths
        ), "Number of images and audio files must match."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load audio
        audio_path = self.audio_paths[self.image_files[idx]]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.audio_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.audio_sample_rate
            )(waveform)

        # Ensure audio is a consistent length by truncating or padding to max_audio_samples
        if waveform.size(1) > self.max_audio_samples:
            waveform = waveform[:, : self.max_audio_samples]
        elif waveform.size(1) < self.max_audio_samples:
            padding = self.max_audio_samples - waveform.size(1)
            waveform = F.pad(waveform, (0, padding))

        return image, waveform
