import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, feed_forward_expansion, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()

        self.mha = nn.MultiheadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feed_forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(feed_forward_expansion * d_model, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm2(x)

        return x


class ConformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=144,
        n_head=4,
        num_layers=4,
        feed_forward_expansion=4,
        dropout_rate=0.1,
    ):
        super(ConformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, n_head, feed_forward_expansion, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = x.transpose(0, 1)

        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)

        x = x.mean(dim=0)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
