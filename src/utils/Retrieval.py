from os.path import isdir, isfile, exists
import torch
import torch.nn as nn
import os
import pathlib
from torch import Tensor
from typing import Tuple, Union


class FetchSimilar:
    def __init__(
        self,
        chkpt_path: str,
        image_path: Union[str, Tuple[str]],
        audio_path: Union[str, Tuple[str]],
        device: str = "cuda",
    ):
        from src.trainer.MultiModalFusionTrainer import MultiModalFusionTrainer

        assert (
            type(image_path) == str or type(image_path) == tuple
        ), "Image path must be a string or a tuple"
        assert (
            type(audio_path) == str or type(audio_path) == tuple
        ), "Audio path must be a string or a tuple"
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'"
        assert isfile(chkpt_path), f"Checkpoint path {chkpt_path} does not exist"

        self.model = MultiModalFusionTrainer.load_from_checkpoint(chkpt_path)
        self.model = self.model.model.to(device)
        self.model.eval()
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.images = {}
        self.audios = {}
        if type(image_path) == str:
            assert isdir(image_path), f"Image path {image_path} does not exist"
            filetypes = ("*.jpeg", "*.jpg", "*.png", "*/*.jpeg", "*/*.jpg", "*/*.png")
            for ft in filetypes:
                for filepath in sorted(pathlib.Path(image_path).glob(ft)):
                    self.images[filepath.stem] = {}
                    self.images[filepath.stem]["path"] = filepath
                    self.images[filepath.stem]["class"] = str(filepath).split(os.sep)[
                        -2
                    ]
        else:
            for img_path in image_path:
                assert isfile(img_path), f"Image path {img_path} does not exist"
                self.images[img_path.stem] = {}
                self.images[img_path.stem]["path"] = img_path
                self.images[img_path.stem]["class"] = str(img_path).split(os.sep)[-2]
        if type(audio_path) == str:
            assert isdir(audio_path), f"Audio path {audio_path} does not exist"
            filetypes = ("*.wav", "*/*.wav")
            for ft in filetypes:
                for filepath in sorted(pathlib.Path(audio_path).glob(ft)):
                    self.audios[filepath.stem] = {}
                    self.audios[filepath.stem]["path"] = filepath
                    self.audios[filepath.stem]["class"] = str(filepath).split(os.sep)[
                        -2
                    ]
        else:
            for aud_path in audio_path:
                assert isfile(aud_path), f"Audio path {aud_path} does not exist"
                self.audios[aud_path.stem] = {}
                self.audios[aud_path.stem]["path"] = aud_path
                self.audios[aud_path.stem]["class"] = str(aud_path).split(os.sep)[-2]

        self.__encode_images()
        self.__encode_audios()

    def __encode_images(self, image_path: str | None = None) -> Tensor | None:
        image_tensor = None
        if image_path is not None:
            assert exists(image_path), f"Image path {image_path} does not exist"
            assert isfile(image_path), f"Image path {image_path} is not a file"
            with torch.inference_mode():
                return self.model.encode_image(image_path, self.device)
        for key, value in self.images.items():
            image_tensor = self.__encode_images(value["path"])
            self.images[key]["embed"] = image_tensor

    def __encode_audios(self, audio_path: str | None = None) -> Tensor | None:
        audio_tensor = None
        if audio_path is not None:
            assert exists(audio_path), f"Audio path {audio_path} does not exist"
            assert isfile(audio_path), f"Audio path {audio_path} is not a file"
            with torch.inference_mode():
                return self.model.encode_speech(audio_path, self.device)
        for key, value in self.audios.items():
            audio_tensor = self.__encode_audios(value["path"])
            self.audios[key]["embed"] = audio_tensor

    def top_k(
        self, path: str | pathlib.PosixPath, modality: str = "all", k: int = 5
    ) -> Tuple[dict, dict]:
        """
        Retrieve the top k most similar items to the given file.

        Args:
            path (str | pathlib.PosixPath): Path to the query file.
            modality (str): Modality to search in ("image", "audio", "all"). Default is "all".
            k (int): Number of top similar items to retrieve. Default is 5.

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:
                - top_k: Dictionary of top k similar items with their similarity scores.
                - query_info: Dictionary containing information about the query file.
        """
        assert modality in ["image", "audio", "all"], "Invalid modality"
        assert isfile(path), f"Path {path} does not exist"
        path = pathlib.Path(path) if type(path) == str else path
        assert path.suffix in [".jpeg", ".jpg", ".png", ".wav"], "Invalid file type"
        scores = {}

        query_class = str(path).split(os.sep)[-2]
        if path.suffix in [".jpeg", ".jpg", ".png"]:
            if path.stem not in self.images.keys():
                query = self.__encode_images(path)
            else:
                query = self.images[path.stem]["embed"]
        else:
            if path.stem not in self.audios.keys():
                query = self.__encode_audios(path)
            else:
                query = self.audios[path.stem]["embed"]
        query_info = {"embed": query, "path": path, "class": query_class}

        if modality == "image" or modality == "all":
            for key, value in self.images.items():
                scores[key + "#image"] = self.cosine_similarity(query, value["embed"])
        if modality == "audio" or modality == "all":
            for key, value in self.audios.items():
                scores[key + "#audio"] = self.cosine_similarity(query, value["embed"])
        scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        top_k = {}
        while len(top_k) < k:
            key, value = scores.popitem()
            filename, md = key.split("#")
            if not (md == modality or modality == "all"):
                continue
            if md == "image":
                top_k[key] = self.images[filename]
                top_k[key]["similarity_score"] = value
            if md == "audio":
                top_k[key] = self.audios[filename]
                top_k[key]["similarity_score"] = value
        return top_k, query_info


def mutualRetrieval(
    score_per_A: torch.Tensor,
    score_per_B: torch.Tensor,
    AB_answers: torch.Tensor,
    BA_answers: torch.Tensor,
    recall_at: list,
    modality_A_title: str = "audio",
    modality_B_title: str = "image",
) -> Tuple[dict, dict, dict]:
    """mutualRetrieval
    A to B and B to A retrieval


    Args:
        score_per_A (torch.Tensor): tensor shape = ( #modalityA_samples, #modalityB)
        score_per_B (torch.Tensor): tensor shape = ( #modalityB, #modalityA_samples)
        AB_answers (torch.Tensor): tensor shape = ( #modalityA_samples,) : list of the golden answer (pair ID) for each instance of madailty A
        BA_answers (torch.Tensor): tensor shape = ( #modalityB_samples,) : list of the golden answer (pair ID) for each instance of madailty B
        modality_A_title (str): the name for modality A
        modality_B_title (str): the name for modality B

    Return:
        Tuple( dict, dict) : recall_results_AB, recall_results_BA, recall_results_mean
    """

    assert len(score_per_A.shape) == 2
    assert len(score_per_B.shape) == 2
    assert len(AB_answers.shape) == 1
    assert len(BA_answers.shape) == 1

    assert score_per_A.shape == (
        len(AB_answers),
        len(BA_answers),
    ), "{} , {}".format(score_per_A.shape, (len(AB_answers), len(BA_answers)))
    assert score_per_B.shape == (
        len(BA_answers),
        len(AB_answers),
    ), "{} , {}".format(score_per_B.shape, (len(BA_answers), len(AB_answers)))

    score_per_A = torch.argsort(score_per_A, dim=1, descending=True).cpu()
    score_per_B = torch.argsort(score_per_B, dim=1, descending=True).cpu()

    # AB : A -> B, BA: B -> A
    rank_AB = BA_answers.reshape(1, -1).repeat(AB_answers.shape[0], 1)
    rank_BA = AB_answers.reshape(1, -1).repeat(BA_answers.shape[0], 1)

    assert rank_AB.shape == score_per_A.shape, (
        rank_AB.shape,
        score_per_A.shape,
    )
    assert rank_BA.shape == score_per_B.shape, (
        rank_BA.shape,
        score_per_B.shape,
    )

    for r in range(AB_answers.shape[0]):
        rank_AB[r, :] = rank_AB[r, score_per_A[r, :]]

    for r in range(BA_answers.shape[0]):
        rank_BA[r, :] = rank_BA[r, score_per_B[r, :]]

    rank_AB = rank_AB == AB_answers.unsqueeze(-1)
    rank_BA = rank_BA == BA_answers.unsqueeze(-1)

    recall_results_AB = {}
    recall_results_BA = {}
    recall_results_mean = {}

    # AB (A to B)
    for k in recall_at:
        if k > rank_AB.shape[1]:
            print(
                "recall@{} is not eligible for #{} {} samples".format(
                    k, rank_AB.shape[1], modality_B_title
                )
            )
        recall_results_AB["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_AB[:, : min(k, rank_AB.shape[1])].reshape(
                        rank_AB.shape[0], min(k, rank_AB.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_AB.shape[0]
        ).item()

    # BBA (B to A)
    for k in recall_at:
        if k > rank_BA.shape[1]:
            print(
                "recall@{} is not eligible for #{} {} samples".format(
                    k, rank_BA.shape[1], modality_A_title
                )
            )
        recall_results_BA["recall@{}".format(k)] = (
            torch.sum(
                torch.max(
                    rank_BA[:, : min(k, rank_BA.shape[1])].reshape(
                        rank_BA.shape[0], min(k, rank_BA.shape[1])
                    ),
                    dim=1,
                    keepdim=True,
                )[0]
            )
            / rank_BA.shape[0]
        ).item()

    for _k in ["recall@{}".format(r) for r in recall_at]:
        recall_results_BA[_k] *= 100
        recall_results_AB[_k] *= 100
        recall_results_mean[_k] = (recall_results_BA[_k] + recall_results_AB[_k]) / 2.0

    return recall_results_AB, recall_results_BA, recall_results_mean
