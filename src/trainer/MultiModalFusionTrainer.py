import platform
from typing import Any

import lightning as L
import torch
from torch import nn, optim

# import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from src.models.MultiModalFusion import MultiModalFusion
from src.losses.ContrastiveLoss import CLIPLoss1D
from src.utils.Retrieval import mutualRetrieval


# TODO: maybe need to explicit clarify what inside config for the sake of save_hyperparameters()
class MultiModalFusionTrainer(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiModalFusion(**config["model_params"])
        if platform.system() == "Linux" and config["compile_model"]:
            # torch.compile requires Triton but currently Triton only supported Linux
            self.model = torch.compile(self.model)
        self.criterion = CLIPLoss1D()
        self.config = config
        self.recall_at = config["recall_at"]
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, loss_info = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {
                "train_loss": loss_info["loss"],
                "train_mean_similarity": loss_info["mean_similarity"],
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["dataloader_params"]["batch_size"],
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, loss_info = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {
                "val_loss": loss_info["loss"],
                "val_mean_similarity": loss_info["mean_similarity"],
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["dataloader_params"]["batch_size"],
        )
        self.validation_step_outputs.append(loss_info)
        return loss_info["mean_similarity"]

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        _, loss_info = self.__get_preds_loss_accuracy(batch)

        self.log_dict(
            {
                "test_loss": loss_info["loss"],
                "test_mean_similarity": loss_info["mean_similarity"],
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["dataloader_params"]["batch_size"],
        )
        self.test_step_outputs.append(loss_info)
        return loss_info["mean_similarity"]

    def on_validation_epoch_end(self) -> None:
        self.calculate_recall_k(self.validation_step_outputs, on_step="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.calculate_recall_k(self.test_step_outputs, on_step="test")
        self.test_step_outputs.clear()

    def __get_preds_loss_accuracy(self, batch):
        image, audio, id = batch["image"], batch["audio"], batch["id"]
        output = self.model(image, audio)
        loss, loss_info = self.criterion(output["image_embed"], output["audio_embed"])
        loss_info["image_embed"] = output["image_embed"]
        loss_info["audio_embed"] = output["audio_embed"]
        loss_info["id"] = id

        return loss, loss_info

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config["optimizer"].upper() == "ADAM":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                betas=self.config["betas"],
                eps=self.config["eps"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
            )

        if self.config["lr_scheduler"].upper() == "EXPONENTIALLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config["gamma"]
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config["step_size"],
                gamma=self.config["gamma"],
            )
        return [optimizer], [scheduler]

    def calculate_recall_k(self, step_outputs: list, on_step: str) -> None:
        all_ids = torch.cat([x["id"] for x in step_outputs], dim=0)
        all_imgs = torch.cat([x["image_embed"] for x in step_outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_audo_feats = torch.cat([x["audio_embed"] for x in step_outputs], dim=0)
        all_audo_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys())).to(self.device)

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats), len(all_audo_feats)
            )
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float().to(self.device),
            all_img_feats.float().T.to(self.device),
        )
        score_per_image = score_per_audio.T

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
            on_step=on_step,
        )

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        on_step: str,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        """reportRetrieval

        Args:
            score_per_A (torch.Tensor): the similarity score per modality A sample
            score_per_B (torch.Tensor): the similarity score per modality B sample
            AB_answers (torch.Tensor): the golden answer (pair ID) for each audio sample
            BA_answers (torch.Tensor): the golden answer (pair ID) for each image sample
            metadata (dict): metadata should include modality the title for A, B and the abbreviation for A and B
        """

        # metadata should include modality the title for A, B and the abbreviation for A and B
        assert "modality_A_title" in metadata
        assert "modality_B_title" in metadata
        assert "modality_A_logAbbr" in metadata
        assert "modality_B_logAbbr" in metadata

        recall_results_AB, recall_results_BA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            recall_at=self.recall_at,
            modality_A_title=metadata["modality_A_title"],
            modality_B_title=metadata["modality_B_title"],
        )

        log_AB_abbr = "{}{}".format(
            metadata["modality_A_logAbbr"], metadata["modality_B_logAbbr"]
        )
        log_BA_abbr = "{}{}".format(
            metadata["modality_B_logAbbr"], metadata["modality_A_logAbbr"]
        )

        print(f"{on_step}_recall_{log_AB_abbr}", recall_results_AB)
        print(f"{on_step}_recall_{log_BA_abbr}", recall_results_BA)
        print(f"{on_step}_recall_mean", recall_results_mean)

        if isinstance(self.logger, WandbLogger):
            # when using wandb
            # self.log(val_recall_AI, {'recall@1': 0.0, 'recall@5': 12.5, 'recall@10': 25.0})
            columns = [f"recall@{i}" for i in self.recall_at]
            dataAB = [[recall_results_AB[col] for col in columns]]
            dataBA = [[recall_results_BA[col] for col in columns]]
            dataMean = [[recall_results_mean[col] for col in columns]]
            self.logger.log_table(
                key="{}_recall_{}".format(on_step, log_AB_abbr),
                columns=columns,
                data=dataAB,
            )
            self.logger.log_table(
                key="{}_recall_{}".format(on_step, log_BA_abbr),
                columns=columns,
                data=dataBA,
            )
            self.logger.log_table(
                key="{}_recall_mean".format(on_step), columns=columns, data=dataMean
            )
        elif isinstance(self.logger, TensorBoardLogger):
            # when using tensorboard
            self.logger.experiment.add_scalars(
                f"{on_step}_recall_{log_AB_abbr}", recall_results_AB, self.global_step
            )
            self.logger.experiment.add_scalars(
                f"{on_step}_recall_{log_BA_abbr}", recall_results_BA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "{on_step}_recall_mean", recall_results_mean, self.global_step
            )
        if self.logger is not None:
            self.log(
                f"{on_step}_recall_mean_10",
                recall_results_mean["recall@10"],
                sync_dist=True,
            )
