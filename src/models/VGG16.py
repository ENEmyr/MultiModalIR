from torch import nn, Tensor
from torchvision import models


class VGG16(nn.Module):
    def __init__(
        self,
        use_pretrained: bool = True,
        freeze_pretrained: bool = True,
        latent_dim: int = 512,
        dropout_rate: float = 0.5,
        num_classes: int = 8,
        **kwargs
    ) -> None:
        super(VGG16, self).__init__()
        if use_pretrained:
            self.model = models.vgg16_bn(weights="DEFAULT")
            if freeze_pretrained:
                for param in self.model.features.parameters():
                    param.requires_grad = False
        else:
            self.model = models.vgg16_bn()
        num_features = self.model.classifier[6].in_features
        feature_layers = list(self.model.classifier.children())[:-1]
        feature_layers.extend(
            [
                nn.Linear(num_features, latent_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(latent_dim, num_classes),
            ]
        )
        self.model.classifier = nn.Sequential(*feature_layers)

    def forward(self, x) -> Tensor:
        return self.model(x)
