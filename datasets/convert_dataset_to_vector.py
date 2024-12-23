import os
import torch
import torchaudio
import random
from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel,
    Wav2Vec2Processor,
    Wav2Vec2ConformerModel,
)

torch.set_float32_matmul_precision("high")


def speech_commands_dataset():
    image_embedding = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    speech_embedding = Wav2Vec2ConformerModel.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )
    speech_processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )

    image_embedding.eval()
    speech_embedding.eval()
    image_embedding = image_embedding.to("cuda")
    speech_embedding = speech_embedding.to("cuda")

    dataset_dir = "./speech-handsign_commands_vectors"
    split_path = ["train", "test", "val"]
    classes = ["go", "no", "up", "yes", "down", "left", "right", "stop"]
    raw_dataset_dir = "./speech-handsign_commands_balanced2"

    if not os.path.exists(dataset_dir):
        for split in split_path:
            for class_ in classes:
                os.makedirs(f"{dataset_dir}/{split}/speech/{class_}")
                os.makedirs(f"{dataset_dir}/{split}/handsign/{class_}")

    with torch.inference_mode():
        latest_embed = None
        counter = 0
        for classname in os.listdir(raw_dataset_dir + "/handsign"):
            counter = 0
            for filename in os.listdir(
                os.path.join(raw_dataset_dir + "/handsign", classname)
            ):
                if filename.split(".")[1] not in ["png", "jpg", "jpeg"]:
                    continue
                filepath = os.path.join(
                    raw_dataset_dir + "/handsign", classname, filename
                )
                img = Image.open(filepath).convert("RGB")
                inputs = image_processor(images=img, return_tensors="pt")
                image_embeded = image_embedding.get_image_features(
                    **inputs.to("cuda")
                )  # inputs['pixel_values'].to("cuda"))
                image_embeded = image_embeded.detach().cpu()
                if counter < 50:
                    torch.save(
                        image_embeded,
                        f"{dataset_dir}/train/handsign/{classname}/{filename.split('.')[0]}.pt",
                    )
                if counter >= 50 and counter < 64:
                    torch.save(
                        image_embeded,
                        f"{dataset_dir}/test/handsign/{classname}/{filename.split('.')[0]}.pt",
                    )
                if counter >= 64:
                    torch.save(
                        image_embeded,
                        f"{dataset_dir}/val/handsign/{classname}/{filename.split('.')[0]}.pt",
                    )
                counter += 1
                latest_embed = image_embeded
        print(
            "DONE :: CONVERT HANDSIGN DATASET TO VECTOR WITH SHAPE", latest_embed.shape
        )

        counter = 0
        for classname in os.listdir(raw_dataset_dir + "/speech"):
            counter = 0
            for filename in os.listdir(
                os.path.join(raw_dataset_dir + "/speech", classname)
            ):
                if filename.split(".")[1] != "wav":
                    continue
                filepath = os.path.join(
                    raw_dataset_dir + "/speech", classname, filename
                )
                speech, sr = torchaudio.load(filepath)
                if sr != 16000:
                    speech = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=16000
                    )(speech)
                inputs = speech_processor(speech, sampling_rate=sr, return_tensors="pt")
                speech_embeded = speech_embedding(
                    inputs["input_values"].squeeze(0).to("cuda")
                ).last_hidden_state.mean(dim=1)
                speech_embeded = speech_embeded.detach().cpu()
                if counter < 50:
                    torch.save(
                        speech_embeded,
                        f"{dataset_dir}/train/speech/{classname}/{filename.split('.')[0]}.pt",
                    )
                if counter >= 50 and counter < 64:
                    torch.save(
                        speech_embeded,
                        f"{dataset_dir}/test/speech/{classname}/{filename.split('.')[0]}.pt",
                    )
                if counter >= 64:
                    torch.save(
                        speech_embeded,
                        f"{dataset_dir}/val/speech/{classname}/{filename.split('.')[0]}.pt",
                    )
                counter += 1
                latest_embed = speech_embeded

        print("DONE :: CONVERT SPEECH DATASET TO VECTOR WITH SHAPE", latest_embed.shape)


def flickr8k_pair_dataset():
    image_embedding = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    speech_embedding = Wav2Vec2ConformerModel.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )
    speech_processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft"
    )

    image_embedding.eval()
    speech_embedding.eval()
    image_embedding = image_embedding.to("cuda")
    speech_embedding = speech_embedding.to("cuda")

    dataset_dir = "./flickr8k_with_audio_vectors"
    image_dir = f"{dataset_dir}/images"
    audio_dir = f"{dataset_dir}/wavs"
    raw_dataset_dir = "./flickr8k_with_audio/"

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    image_files = []
    audio_paths = {}

    for filename in sorted(os.listdir(raw_dataset_dir + "/images")):
        rand_list = []
        filename_no_ext = filename.split(".")[0]
        if os.path.exists(
            os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_0.wav")
        ):
            rand_list.append(
                os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_0.wav")
            )
        if os.path.exists(
            os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_1.wav")
        ):
            rand_list.append(
                os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_1.wav")
            )
        if os.path.exists(
            os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_2.wav")
        ):
            rand_list.append(
                os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_2.wav")
            )
        if os.path.exists(
            os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_3.wav")
        ):
            rand_list.append(
                os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_3.wav")
            )
        if os.path.exists(
            os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_4.wav")
        ):
            rand_list.append(
                os.path.join(raw_dataset_dir + "/wavs", f"{filename_no_ext}_4.wav")
            )
        if len(rand_list) == 0:
            continue
        image_files.append(filename)
        audio_paths[filename] = random.choice(rand_list)

    tot_number = 1
    with torch.inference_mode():
        latest_img_embed = None
        latest_audio_embed = None
        for filename in image_files:
            filepath = os.path.join(raw_dataset_dir + "/images", filename)
            img = Image.open(filepath).convert("RGB")
            inputs = image_processor(images=img, return_tensors="pt")
            image_embeded = image_embedding.get_image_features(**inputs.to("cuda"))
            image_embeded = image_embeded.detach().cpu()
            torch.save(
                image_embeded,
                f"{image_dir}/{filename.split('.')[0]}.pt",
            )
            latest_img_embed = image_embeded

            wav, sr = torchaudio.load(audio_paths[filename])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
            inputs = speech_processor(wav, sampling_rate=16000, return_tensors="pt")
            speech_embeded = speech_embedding(
                inputs["input_values"].squeeze(0).to("cuda")
            ).last_hidden_state.mean(dim=1)
            speech_embeded = speech_embeded.detach().cpu()
            torch.save(
                speech_embeded,
                f"{audio_dir}/{filename.split('.')[0]}.pt",
            )
            latest_audio_embed = speech_embeded
            tot_number += 1

        print(
            "DONE :: CONVERT IMAGE DATASET TO VECTOR WITH SHAPE", latest_img_embed.shape
        )
        print(
            "DONE :: CONVERT WAV DATASET TO VECTOR WITH SHAPE", latest_audio_embed.shape
        )
        print("TOTAL NUMBER :: ", tot_number)


if __name__ == "__main__":
    speech_commands_dataset()
