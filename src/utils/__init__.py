import torch
from typing import Tuple


def pad_image_embeddings(
    image_embeddings: torch.Tensor, target_embed_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads image embeddings to match the target sequence length.

    :param image_embeddings: Tensor of shape (batch_size, image_embed_dim, embed_dim)
    :param target_embed_dim: Target sequence length to pad to (e.g., audio sequence length)
    :return: Padded image embeddings and attention mask
    """
    batch_size, image_embed_dim = image_embeddings.size()

    # Compute padding length
    pad_length = target_embed_dim - image_embed_dim
    if pad_length > 0:
        # Create padding tensor (zero values)
        padding = torch.zeros(batch_size, pad_length, device=image_embeddings.device)

        # Concatenate padding to image embeddings
        padded_image_embeddings = torch.cat([image_embeddings, padding], dim=1)

        # Create attention mask: 1 for valid positions, 0 for padding
        image_attention_mask = torch.cat(
            [
                torch.ones(batch_size, image_embed_dim, device=image_embeddings.device),
                torch.zeros(batch_size, pad_length, device=image_embeddings.device),
            ],
            dim=1,
        ).long()
    else:
        # No padding needed
        padded_image_embeddings = image_embeddings
        image_attention_mask = torch.ones(
            batch_size, image_embed_dim, device=image_embeddings.device
        ).long()

    return padded_image_embeddings, image_attention_mask
