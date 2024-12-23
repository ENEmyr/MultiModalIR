import os
from typing import Union
import torch
from torch import nn
from torch.nn import functional as F
from src.models.utils.layers import get_projection
from src.models.utils.fusion_transformer import FusionTransformer
from timm.layers import trunc_normal_
from pathlib import PosixPath


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        image_embed_dim,
        audio_embed_dim,
        fusion_params,
        image_max_tokens=None,
        audio_max_tokens=None,
        projection_dim=6144,
        token_projection="gated",
        projection="gated",
        individual_projections=True,
        use_positional_emb=False,
    ):
        super().__init__()

        self.fusion = FusionTransformer(**fusion_params)

        self.individual_projections = individual_projections
        self.use_positional_emb = use_positional_emb

        self.embed_dim = fusion_params["embed_dim"]

        self.image_norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.audio_norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)

        if self.use_positional_emb:
            assert image_max_tokens is not None
            assert audio_max_tokens is not None
            self.image_pos_embed = nn.Parameter(
                torch.zeros(1, image_max_tokens, self.embed_dim)
            )
            self.audio_pos_embed = nn.Parameter(
                torch.zeros(1, audio_max_tokens, self.embed_dim)
            )
        else:
            self.image_pos_embed = None
            self.audio_pos_embed = None

        self.image_token_proj = get_projection(
            image_embed_dim, self.embed_dim, token_projection
        )
        self.audio_token_proj = get_projection(
            audio_embed_dim, self.embed_dim, token_projection
        )

        if not self.individual_projections:
            self.proj = get_projection(self.embed_dim, projection_dim, projection)
        else:
            self.image_proj = get_projection(self.embed_dim, projection_dim, projection)
            self.audio_proj = get_projection(self.embed_dim, projection_dim, projection)

        self.image_embedding = None
        self.image_processor = None
        self.speech_embedding = None
        self.speech_processor = None

        self.init_weights()

    def init_weights(self):
        for weights in [
            self.image_pos_embed,
            self.audio_pos_embed,
        ]:
            if weights is not None:
                trunc_normal_(weights, std=0.02)

    def _check_and_fix_if_input_empty(self, x, attention_mask):
        nonempty_input_mask = attention_mask.sum(-1) != 0

        # if all tokens of modality is empty, add one masking token
        empty_input_mask = nonempty_input_mask == 0
        n_masking_tokens = 1
        x[empty_input_mask, :n_masking_tokens] = self.fusion.masking_token.type(x.dtype)
        attention_mask[empty_input_mask, :n_masking_tokens] = 1
        return x, attention_mask, nonempty_input_mask

    def extract_image_tokens(self, image, attention_mask):
        x = self.image_token_proj(image)
        x = self.image_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(
            x, attention_mask
        )
        special_token_mask = attention_mask == 0

        return {
            "all_tokens": x,
            "attention_mask": attention_mask,
            "special_token_mask": special_token_mask,
            "nonempty_input_mask": nonempty_input_mask,
        }

    def extract_audio_tokens(self, audio, attention_mask):
        x = self.audio_token_proj(audio)
        x = self.audio_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(
            x, attention_mask
        )
        special_token_mask = attention_mask == 0
        return {
            "all_tokens": x,
            "attention_mask": attention_mask,
            "special_token_mask": special_token_mask,
            "nonempty_input_mask": nonempty_input_mask,
        }

    def forward(self, image_embed, audio_embed):
        output = {}

        if image_embed["attention_mask"].dim() > 2:
            # [batch_size, 1, image_max_tokens] -> [batch_size, image_max_tokens]
            image_embed["attention_mask"] = image_embed["attention_mask"].squeeze(1)
        if audio_embed["attention_mask"].dim() > 2:
            # [batch_size, 1, audio_max_tokens] -> [batch_size, audio_max_tokens]
            audio_embed["attention_mask"] = audio_embed["attention_mask"].squeeze(1)

        image_raw_embed = self.extract_image_tokens(
            image_embed["embed"], image_embed["attention_mask"]
        )
        audio_raw_embed = self.extract_audio_tokens(
            audio_embed["embed"], audio_embed["attention_mask"]
        )
        output["image_nonempty_input_mask"] = image_raw_embed["nonempty_input_mask"]
        output["audio_nonempty_input_mask"] = audio_raw_embed["nonempty_input_mask"]

        # add positional embedding after masking
        if self.use_positional_emb:
            audio_raw_embed["all_tokens"] = (
                audio_raw_embed["all_tokens"] + self.audio_pos_embed
            )
            image_raw_embed["all_tokens"] = (
                image_raw_embed["all_tokens"] + self.image_pos_embed
            )

        audio = self.fusion(audio=audio_raw_embed)["audio"]
        image = self.fusion(image=image_raw_embed)["image"]

        if not self.individual_projections:
            output["audio_embed"] = self.proj(audio["embed"])
            output["image_embed"] = self.proj(image["embed"])
        else:
            output["audio_embed"] = self.audio_proj(audio["embed"])
            output["image_embed"] = self.image_proj(image["embed"])

        return output

    def encode_image(
        self, image: Union[Union[str, PosixPath], torch.Tensor], device: str = "cuda"
    ) -> torch.Tensor:
        image_tensor = None if type(image) == str or type(image) == PosixPath else image
        with torch.inference_mode():
            if image_tensor is None:
                assert os.path.exists(image), f"Image path {image} does not exist"
                from PIL import Image

                if self.image_embedding is None or self.image_processor is None:
                    from transformers import (
                        CLIPProcessor,
                        CLIPModel,
                    )

                    self.image_embedding = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                    self.image_processor = CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                    self.image_embedding.eval()
                    self.image_embedding.to(device)
                image_tensor = self.image_processor(
                    images=Image.open(image).convert("RGB"), return_tensors="pt"
                )
                image_tensor = self.image_embedding.get_image_features(
                    **image_tensor.to(device)
                )

            # Compute padding length
            pad_length = self.embed_dim - image_tensor.size(1)
            if pad_length > 0:
                # Create padding tensor (zero values)
                padding = torch.zeros(1, pad_length, device=device)
                # Concatenate padding to image embeddings
                image_tensor = torch.cat([image_tensor, padding], dim=1)

            image_raw_embed = self.extract_image_tokens(
                image_tensor.unsqueeze(0),
                torch.ones(1, image_tensor.size(1), device=device).long(),
            )

            embed = self.fusion(image=image_raw_embed)["image"]["embed"]
            if not self.individual_projections:
                embed = self.proj(embed)
            else:
                embed = self.image_proj(embed)
        return embed.squeeze().detach()

    def encode_speech(
        self, audio: Union[Union[str, PosixPath], torch.Tensor], device: str = "cuda"
    ) -> torch.Tensor:
        audio_tensor = None if type(audio) == str or type(audio) == PosixPath else audio
        with torch.inference_mode():
            if audio_tensor is None:
                assert os.path.exists(audio), f"Audio path {audio} does not exist"
                import torchaudio

                if self.speech_embedding is None or self.speech_processor is None:
                    from transformers import (
                        Wav2Vec2Processor,
                        Wav2Vec2ConformerModel,
                    )

                    self.speech_embedding = Wav2Vec2ConformerModel.from_pretrained(
                        "facebook/wav2vec2-conformer-rope-large-960h-ft"
                    )
                    self.speech_processor = Wav2Vec2Processor.from_pretrained(
                        "facebook/wav2vec2-conformer-rope-large-960h-ft"
                    )
                    self.speech_embedding.eval()
                    self.speech_embedding.to(device)
                speech, sr = torchaudio.load(audio)
                if sr != 16000:
                    speech = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=16000
                    )(speech)
                inputs = self.speech_processor(
                    speech, sampling_rate=sr, return_tensors="pt"
                )
                audio_tensor = self.speech_embedding(
                    inputs["input_values"].squeeze(0).to(device)
                ).last_hidden_state.mean(dim=1)
            audio_raw_embed = self.extract_audio_tokens(
                audio_tensor.unsqueeze(0),
                torch.ones(1, audio_tensor.size(1), device=device).long(),
            )
            embed = self.fusion(audio=audio_raw_embed)["audio"]["embed"]
            if not self.individual_projections:
                embed = self.proj(embed)
            else:
                embed = self.audio_proj(embed)
        return embed.squeeze().detach()
