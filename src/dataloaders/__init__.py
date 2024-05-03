from typing import Dict, Tuple
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import DataLoader


class CustomDataloader(ABC):
    def __init__(
        self, dataset_split_paths: Dict[str, str], config: dict, verbose: bool
    ) -> None:
        self.dataset_split_paths = dataset_split_paths
        self.config = config
        self.verbose = verbose

    @property
    @abstractmethod
    def train(self) -> DataLoader:
        pass

    @property
    @abstractmethod
    def test(self) -> DataLoader:
        pass

    @property
    @abstractmethod
    def validate(self) -> DataLoader:
        pass

    @abstractmethod
    def _collate_fn(self, batch) -> Tuple:
        pass

    @abstractmethod
    def _verbose(self) -> None:
        pass
