import os
import pathlib
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot, pad
from typing import Tuple, List, Dict
from src.utils.TextTransform import TextTransform
from src.utils import pad_image_embeddings


class MiniSpeechHandsignVector(Dataset):
    def __init__(
        self, target_dir: str, padding: str = "none", text_transform: bool = False
    ) -> None:
        self.handsign_paths = sorted(
            list(pathlib.Path(target_dir).glob("handsign/*/*.pt"))
        )
        self.speech_paths = sorted(list(pathlib.Path(target_dir).glob("speech/*/*.pt")))
        self.paths = list(zip(self.handsign_paths, self.speech_paths))
        self.target_dir = target_dir
        self.classes, self.class_to_idx = self.__find_class()
        self.text_transform = text_transform
        self.padding = padding
        self.id2Filename = [x.stem for x in self.handsign_paths]
        self.filename2Id = {x: i for i, x in enumerate(self.id2Filename)}
        assert padding in ["none", "zero"], "Padding must be 'none' or 'zero'."

    def __find_class(self) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            [entry.name for entry in list(os.scandir(self.target_dir + "/handsign/"))]
        )
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        return classes, class_to_idx

    def __load_vector(
        self, index: int
    ) -> Dict[Dict[torch.Tensor, torch.Tensor], Dict[torch.Tensor, torch.Tensor]]:
        vector = dict()
        hs_path, sp_path = self.paths[index]
        hs_vector = torch.load(hs_path)
        sp_vector = torch.load(sp_path)
        batch_size = hs_vector.size(0)
        if self.padding.lower() != "none":
            padd_image, image_attention_mask = pad_image_embeddings(
                hs_vector, sp_vector.shape[1]
            )
            vector["image"] = {
                # "embed": hs_vector.unsqueeze(0),
                "embed": padd_image,
                "attention_mask": image_attention_mask,
            }
        else:
            vector["image"] = {
                # "embed": hs_vector.unsqueeze(0),
                "embed": hs_vector,
                "attention_mask": torch.ones(
                    batch_size, hs_vector.shape[0], 1, device=hs_vector.device
                ).long(),
            }
        vector["audio"] = {
            # "embed": sp_vector.unsqueeze(0),
            "embed": sp_vector,
            "attention_mask": torch.ones(
                batch_size, sp_vector.shape[0], 1, device=sp_vector.device
            ).long(),
        }
        return vector

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, index: int
    ) -> Dict[Dict[torch.Tensor, torch.Tensor], Dict[torch.Tensor, torch.Tensor]]:
        vector = self.__load_vector(index)
        class_name = self.paths[index][0].parent.name
        class_idx = self.class_to_idx[class_name]

        label = None
        if self.text_transform:
            tt = TextTransform()
            label = torch.tensor(tt.text_to_int(class_name.upper()))
        else:
            label = one_hot(
                torch.tensor([class_idx]), num_classes=len(self.class_to_idx)
            )
        vector["label"] = label
        vector["id"] = self.filename2Id[self.paths[index][0].stem]

        return vector

    def decode_label(self, pred: int | torch.Tensor) -> str:
        if isinstance(pred, torch.Tensor):
            pred = int(pred.argmax(pred.ndim - 1).item())
        return self.classes[pred]
