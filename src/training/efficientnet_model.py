import torch.nn as nn
import torchvision.models as models


class EfficientNetClassifier(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        return self.backbone(x)