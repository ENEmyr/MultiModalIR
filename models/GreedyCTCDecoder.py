import torch
from typing import List
from utils.TextTransform import TextTransform


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = TextTransform()

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        strs = [self.transform.int_to_text(s.tolist()) for s in indices]
        return strs
