import torch
import os
import argparse
import json
import lightning as L
from rich import console
from src.utils.Callback import VerboseCallback

console = console.Console()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    help="Path to config file",
    type=str,
    dest="conf_path",
    required=True,
)

parser.add_argument(
    "-w",
    "--use-wandb",
    help="Use Wandb as a tracker or not",
    action="store_true",
    dest="use_wandb",
    default=False,
)
TRAIN = "train"
VAL = "val"
TEST = "test"


def Wav2VecConformerClsTrainer(trainer: L.Trainer, config: dict):
    from transformers import AutoProcessor
    from src.dataloaders.speech import MiniSpeechCommands
    from src.trainer.Wav2Vec2ConformerTrainer import Wav2Vec2ConformerTrainer

    transform = AutoProcessor.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft", return_attention_mask=False
    )

    def collate_fn(batch):
        inputs = []
        labels = []
        for input, label in batch:
            inputs.append(input.T)
            labels.append(label)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels = torch.stack(labels)
        return inputs.squeeze(), labels.squeeze().to(torch.float32)

    data_transform = {TRAIN: transform, VAL: transform, TEST: transform}

    speech_datasets = {
        x: MiniSpeechCommands(
            os.path.join(config["dataset_path"], x),
            transform=data_transform[x],
            text_transform=False,
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            speech_datasets[x],
            batch_size=config["batch_size"],
            shuffle=(x == TRAIN),
            num_workers=config["num_workers"],
            collate_fn=collate_fn,
        )  # os.cpu_count() = 24
        for x in [TRAIN, VAL, TEST]
    }

    dataset_sizes = {x: len(speech_datasets[x]) for x in [TRAIN, VAL, TEST]}
    class_names = speech_datasets[TRAIN].classes

    for x in [TRAIN, VAL, TEST]:
        console.log("Loaded {} audios under {}".format(dataset_sizes[x], x))
    console.log("Classes: ", class_names)

    model = Wav2Vec2ConformerTrainer(config)
    trainer.fit(
        model=model,
        train_dataloaders=dataloaders[TRAIN],
        val_dataloaders=dataloaders[VAL],
    )
    trainer.test(model=model, dataloaders=dataloaders[TEST], verbose=True)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.isfile(args.conf_path) and args.conf_path.split(".")[-1] == "json"
    with open(args.conf_path, "r") as f:
        config = json.load(f)

    assert config["model"] in ("Wav2Vec2ConformerCls", "Vgg16", "MultiFusion")
    assert os.path.exists(config["dataset_path"])
    assert os.path.exists(config["weight_save_path"])

    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["weight_save_path"],
        benchmark=config["benchmark"],
        precision=config["precision"],
        max_epochs=config["epochs"],
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=config["log_frequency"],
        # callbacks=VerboseCallback(),
    )

    if config["model"] == "Wav2Vec2ConformerCls":
        Wav2VecConformerClsTrainer(trainer, config)
    else:
        pass
