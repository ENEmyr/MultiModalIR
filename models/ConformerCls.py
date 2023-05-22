import torch
import torch.nn as nn
from torchaudio.models import Conformer


class ConvolutionSubsampling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(ConvolutionSubsampling, self).__init__()
        self.conv_subsample = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_subsample(x.unsqueeze(1))  # (batch_size, 1, time, d_model)
        batch_size, d_model, subsampled_time, subsampled_freq = output.size()
        output = output.permute(
            0, 2, 1, 3
        )  # batch_size, subsampled_time, d_model, subsampled_freq
        output = output.contiguous().view(
            batch_size, subsampled_time, d_model * subsampled_freq
        )
        return output


class ConformerCls(nn.Module):
    def __init__(
        self,
        d_input: int = 80,
        d_model: int = 144,
        dropout: float = 0.1,
        num_heads: int = 4,
        ffn_expansion_factor: int = 4,
        depthwise_conv_kernel_size: int = 31,
        num_conformer_layers: int = 16,
        num_lstm_layers: int = 1,
        num_lstm_hidden: int = 320,
        num_classes: int = 8,
    ) -> None:
        super(ConformerCls, self).__init__()
        self.conv_subsample = ConvolutionSubsampling(d_model)
        self.linear_proj = nn.Linear(
            in_features=(d_model * (((d_input - 1) // 2 - 1) // 2)),
            out_features=d_model,
        )
        self.dropout = nn.Dropout(dropout)
        self.conformer_block = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=d_model * ffn_expansion_factor,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            num_layers=num_conformer_layers,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=num_lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=24 * num_lstm_hidden, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.conv_subsample(x)
        x = self.linear_proj(x)
        x = self.dropout(x)
        x, _ = self.conformer_block(x, input_lengths)
        x, _ = self.lstm_decoder(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    conformer_classifier = ConformerCls()
    test_inp = torch.rand((2, 101, 80))
    test_len = torch.tensor([24, 24])
    output = conformer_classifier(test_inp, test_len)
    # _, pred = torch.max(output.data, 1)
    pred = torch.softmax(output, dim=1).argmax(dim=1)
    console.log(output.shape)
    console.log(output)
    console.log(pred)
    console.log(pred.dtype)
