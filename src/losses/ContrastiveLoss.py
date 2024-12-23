import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from src.models.utils.layers import sim_matrix


MAX_EYE = 256


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return -loss_i - loss_j


class MMS_Loss(nn.Module):
    def __init__(self, margin=0.001):
        super(MMS_Loss, self).__init__()
        self.margin = margin

    def forward(
        self,
        S,
    ):
        deltas = self.margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss


class NCELoss(nn.Module):
    def __init__(self, loss_type="NormSoftmax", temperature=0.05, ia_weight=1.0):
        super().__init__()

        if loss_type == "NormSoftmax":
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif loss_type == "MMS":
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.ia_weight = ia_weight

    def forward(self, input_data):

        nonempty = {}
        nonempty["ia"] = (
            input_data["image_nonempty_input_mask"]
            & input_data["audio_nonempty_input_mask"]
        )

        loss_sum = 0
        weight_sum = 0
        loss_info = {}

        for name, embed_name1, embed_name2, weight in [
            ("ia", "image_embed", "audio_embed", self.ia_weight)
        ]:
            if (
                (embed_name1 in input_data)
                and (embed_name2 in input_data)
                and (weight != 0)
            ):
                nonempty_mask = nonempty[name]
                embed1 = input_data[embed_name1][nonempty_mask]
                embed2 = input_data[embed_name2][nonempty_mask]

                loss = self.contrastive_loss(sim_matrix(embed1, embed2))
                loss_sum += weight * loss
                weight_sum += weight
                loss_info["mean_similarity"] = (
                    self.cosine_sim(embed1, embed2).mean().item()
                )

        final_loss = loss_sum / weight_sum
        loss_info["loss"] = final_loss.item()
        return final_loss, loss_info


class NCELoss2(nn.Module):
    def __init__(self, loss_type="NormSoftmax", temperature=0.05, **kwargs):
        super().__init__()
        if loss_type == "NormSoftmax":
            self.loss = NormSoftmaxLoss(temperature)
        elif loss_type == "MMS":
            self.loss = MMS_Loss()
        else:
            raise NotImplementedError()
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, embed1, embed2):
        loss_info = {}
        loss = self.loss(sim_matrix(embed1, embed2))
        loss_info["loss"] = loss.item()
        loss_info["mean_similarity"] = self.cosine_sim(embed1, embed2).mean().item()
        return loss, loss_info


class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_audio = nn.CrossEntropyLoss()

    def forward(self, image_features, audio_features):
        loss_info = {}
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ audio_features.t()
        logits_per_text = logit_scale * audio_features @ image_features.t()

        loss_info["mean_similarity"] = (
            (image_features @ audio_features.t()).max(dim=1)[0].mean().item()
            + (audio_features @ image_features.t()).max(dim=1)[0].mean().item()
        ) / 2

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )
        final_loss = (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_audio(logits_per_text, ground_truth)
        ) / 2
        loss_info["loss"] = final_loss.item()
        return final_loss, loss_info
