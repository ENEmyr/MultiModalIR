import pytest
import torch
import os
import json
import platform

from pathlib import Path
from typing import Tuple
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.trainer.VGG16Trainer import VGG16Trainer


@pytest.fixture
def model():
    m = VGG16Trainer.load_from_checkpoint(
        "./logs/MultiModalFusion/d2m54pbm/checkpoints/epoch=5-val_loss=0.00-val_accuracy=1.00.ckpt"
    )
    # m = m.model
    m = m.to("cuda")
    m.eval()
    return m  # Instantiate the model without pretrained weights


@pytest.fixture
def mock_input_data():
    return torch.randn(2, 3, 224, 224).to("cuda")


@pytest.fixture
def dataset_path():
    with open("./configs/VGG16.json", "r") as f:
        config = json.load(f)
    return config["dataset_path"]


@pytest.fixture
def label_decoder(dataset_path):
    classes = sorted(
        [entry.name for entry in list(os.scandir(os.path.join(dataset_path, "test")))]
    )
    idx_to_class = {idx: c for idx, c in enumerate(classes)}
    return idx_to_class


@pytest.fixture
def transform():
    return transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


@pytest.fixture
def load_input_data(dataset_path, transform):
    def _load_input_data(file_path) -> Tuple[Tensor, str]:
        img_path = Path(os.path.join(dataset_path, file_path))
        label = img_path.parent.name
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img.load()
        img = img.convert("RGB")
        return transform(img).to("cuda"), label.upper()

    return _load_input_data


def test_forward_output_shape(model, mock_input_data):
    # Output shape should match (batch_size, num_classes)
    if platform.system() == "Linux":
        model = torch.compile(model)
    with torch.inference_mode():
        output = model(mock_input_data)
    assert output.shape == torch.Size([2, 8])


@pytest.mark.parametrize(
    "file_path",
    [
        "test/left/hand5_g_bot_seg_2_cropped.jpeg",
        "test/go/hand5_o_dif_seg_2_cropped.jpeg",
        "test/stop/hand5_s_bot_seg_2_cropped.jpeg",
    ],
)
def test_inference(model, label_decoder, load_input_data, file_path):
    input_data, label = load_input_data(file_path)
    if platform.system() == "Linux":
        model = torch.compile(model)
    with torch.inference_mode():
        y_pred = model(input_data.unsqueeze(0))
    y_pred_argmax = torch.argmax(y_pred, dim=1).to(torch.float32)
    label_pred = label_decoder.get(y_pred_argmax[0].item())
    assert label_pred.upper() == label
