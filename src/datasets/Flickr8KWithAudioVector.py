import os
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from src.utils import pad_image_embeddings


class Flickr8KWithAudioVector(Dataset):
    def __init__(self, image_dir, audio_dir, padding: str = "none"):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            audio_dir (str): Path to the directory containing audio files.
        """
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.audio_files = sorted(os.listdir(audio_dir))
        self.padding = padding
        assert len(self.image_files) == len(
            self.audio_files
        ), "Number of images and audio files must match."
        assert padding in ["none", "zero"], "Padding must be 'none' or 'zero'."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = torch.load(os.path.join(self.image_dir, self.image_files[idx]))
        audio = torch.load(os.path.join(self.audio_dir, self.audio_files[idx]))
        batch_size = image.size(0)
        image_embed_dim = image.size(1)
        audio_embed_dim = audio.size(1)
        output = {
            "image": {
                "embed": image,
                "attention_mask": torch.ones(
                    batch_size, image_embed_dim, device=image.device
                ).long(),
            },
            "audio": {
                "embed": audio,
                "attention_mask": torch.ones(
                    batch_size, audio_embed_dim, device=audio.device
                ).long(),
            },
        }
        if self.padding.lower() != "none":
            padd_image, image_attention_mask = pad_image_embeddings(
                image, audio_embed_dim
            )
            output["image"]["embed"] = padd_image
            output["image"]["attention_mask"] = image_attention_mask

        return output
