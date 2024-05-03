from typing import Tuple
import pytest
import torch
import torchaudio
import os
import json

from src.models.Wav2Vec2ConformerCls import Wav2Vec2ConformerCls
from transformers import AutoProcessor
from pathlib import Path
from torch import Tensor

torch.set_float32_matmul_precision("high")


@pytest.fixture
def model(config):
    m = Wav2Vec2ConformerCls(**config)
    m.load_state_dict(torch.load("./weights/Wav2Vec2Conformer.pt"))
    m.eval()
    return m  # Instantiate the model without pretrained weights


@pytest.fixture()
def config():
    with open("./configs/Wav2Vec2ConformerCls.json", "r") as f:
        config = json.load(f)
    return config


@pytest.fixture
def mock_input_data():
    return torch.randn(2, 16000)


@pytest.fixture
def label_decoder(config):
    classes = sorted(
        [
            entry.name
            for entry in list(os.scandir(os.path.join(config["dataset_path"], "test")))
        ]
    )
    idx_to_class = {idx: c for idx, c in enumerate(classes)}
    return idx_to_class


@pytest.fixture
def transform():
    return AutoProcessor.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )


@pytest.fixture
def load_input_data(config, transform):
    def _load_input_data(file_path) -> Tuple[Tensor, str]:
        wav_path = Path(os.path.join(config["dataset_path"], file_path))
        label = wav_path.parent.name
        wav, sr = torchaudio.load(wav_path)
        wav = transform(wav, sampling_rate=sr, return_tensors="pt")
        return wav.input_values, label.upper()

    return _load_input_data


def test_gradients_frozen(model):
    # Check that gradients are frozen for pretrained parameters
    for param in model.wav2vec2conformer.parameters():
        assert not param.requires_grad


def test_forward_output_shape(model, mock_input_data):
    # Output shape should match (batch_size, num_classes)
    with torch.inference_mode():
        output = model(mock_input_data)
    assert output.shape == torch.Size([2, 8])


@pytest.mark.parametrize(
    "file_path",
    [
        "test/stop/ced835d3_nohash_4.wav",
        "test/go/ccb1266b_nohash_0.wav",
        "test/yes/ccf418a5_nohash_0.wav",
    ],
)
def test_inference(model, label_decoder, load_input_data, file_path):
    input_data, label = load_input_data(file_path)
    with torch.inference_mode():
        y_pred = model(input_data.squeeze(0))
    label_pred = label_decoder.get(y_pred.argmax(1).item())
    assert label_pred.upper() == label
