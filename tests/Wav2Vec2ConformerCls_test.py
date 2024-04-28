import pytest
import torch
import torchaudio
import os
from models.Wav2Vec2ConformerCls import Wav2Vec2ConformerCls
from transformers import AutoProcessor


@pytest.fixture
def model():
    m = Wav2Vec2ConformerCls(use_pretrained=True)
    m = m.to("cuda")
    return m  # Instantiate the model without pretrained weights


@pytest.fixture
def mock_input_data():
    return torch.randn(2, 16000).to("cuda")


@pytest.fixture
def dataset_path():
    return "./datasets/speech-handsign_commands/speech/"


@pytest.fixture
def get_label(dataset_path):
    classes = sorted(
        [entry.name for entry in list(os.scandir(os.path.join(dataset_path, "/val")))]
    )
    idx_to_class = {idx: c for idx, c in enumerate(classes)}
    return idx_to_class


@pytest.fixture
def transform():
    return AutoProcessor.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )


@pytest.fixture
def load_input_data(dataset_path, transform):
    def _load_input_data(file_path):
        wav_path = os.path.join(dataset_path, file_path)
        wav, sr = torchaudio.load(wav_path)
        wav = transform(wav.reshape(-1), sampling_rate=sr, return_tensors="pt")
        return wav.input_values

    return _load_input_data


@pytest.mark.parametrize(
    "file_path,label",
    [
        ("/test/stop/ced835d3_nohash_4.wav", "STOP"),
        ("/test/go/ccb1266b_nohash_0.wav", "GO"),
        ("/test/yes/ccf418a5_nohash_0.wav", "YES"),
    ],
)
def test_inference(model, get_label, load_input_data, file_path, label):
    input_data = load_input_data(file_path)
    y_pred = model(input_data)
    label_pred = get_label.get(y_pred.argmax(1).item())
    assert label_pred == label


def test_forward_output_shape(model, mock_input_data):
    output = model(mock_input_data)
    assert output.shape == torch.Size(
        [2, 8]
    )  # Output shape should match (batch_size, sequence_length, num_classes)


def test_gradients_frozen(model):
    for param in model.wav2vec2conformer.parameters():
        assert (
            not param.requires_grad
        )  # Check that gradients are frozen for pretrained parameters


# TODO: implement unit test for a run thought model with mini-batch of real data
