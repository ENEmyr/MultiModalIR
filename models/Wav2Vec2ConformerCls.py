from torch import nn
from torch.functional import Tensor
from transformers import Wav2Vec2ConformerModel, Wav2Vec2ConformerConfig


class Wav2Vec2ConformerCls(nn.Module):
    def __init__(
        self,
        use_pretrained: bool = True,
        freeze_pretrained: bool = True,
        num_classes: int = 8,
        linear_in: int = 1024,
        dropout_rate: float = 0.1,
        wav2vec2conformer_conf: Wav2Vec2ConformerConfig = None,
        **kwargs
    ) -> None:
        super(Wav2Vec2ConformerCls, self).__init__()
        if use_pretrained:
            self.wav2vec2conformer = Wav2Vec2ConformerModel.from_pretrained(
                "facebook/wav2vec2-conformer-rope-large-960h-ft"
            )
            if freeze_pretrained:
                for param in self.wav2vec2conformer.parameters():
                    param.requires_grad = False
        else:
            self.wav2vec2conformer = Wav2Vec2ConformerModel(wav2vec2conformer_conf)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            in_features=linear_in, out_features=num_classes, bias=True
        )

    def forward(self, x) -> Tensor:
        x = self.wav2vec2conformer(input_values=x)
        x = x.last_hidden_state
        assert x.ndim == 3 and x.shape[2] == 1024
        x = self.dropout(x)
        logits = self.classifier(x)
        logits = logits.max(1).values
        assert logits.ndim == 2 and logits.shape[1] == 8  # (batch, num_classes)
        return nn.functional.softmax(logits, 1)
