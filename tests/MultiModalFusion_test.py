import pytest
import torch
import yaml

from src.trainer.MultiModalFusionTrainer import MultiModalFusionTrainer

torch.set_float32_matmul_precision("high")


@pytest.fixture
def model(config):
    m = MultiModalFusionTrainer.load_from_checkpoint(
        "./weights/MultiModalFusion_epoch=31-val_loss=2.23-val_mean_similarity=0.37.ckpt"
    )
    m = m.model.to("cuda")
    m.eval()
    return m  # Instantiate the model without pretrained weights


@pytest.fixture()
def config():
    with open("./configs/MultiModalFusion.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def encode_image(model):
    def _encode_image(image_path):
        return model.encode_image(image_path)

    return _encode_image


@pytest.fixture
def encode_speech(model):
    def _encode_speech(speech_path):
        return model.encode_speech(speech_path)

    return _encode_speech


def test_forward_output_shape(config, encode_image, encode_speech):
    # Output shape should match (projection_dim,)
    image_embed = encode_image(
        "./datasets/speech-handsign_commands_balanced2/handsign/go/go_1.jpeg"
    )
    audio_embed = encode_speech(
        "./datasets/speech-handsign_commands_balanced2/speech/go/go_1.wav"
    )
    assert image_embed.shape == torch.Size([config["model_params"]["projection_dim"]])
    assert audio_embed.shape == torch.Size([config["model_params"]["projection_dim"]])
    assert image_embed.shape == audio_embed.shape
