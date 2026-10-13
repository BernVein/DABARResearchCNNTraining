import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class RawMatCNN(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()

        if num_outputs < 1:
            raise ValueError("num_outputs must be positive")

        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT,
        )
        feature_size = backbone.classifier[0].in_features

        self.encoder = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.head = nn.Sequential(
            nn.Linear(feature_size * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen BatchNorm statistics unchanged.
        self.encoder.eval()
        return self

    def extract_features(self, whole, patches):
        batch_size, patch_count = patches.shape[:2]

        views = torch.cat([whole.unsqueeze(1), patches], dim=1)
        features = self.encoder(views.flatten(0, 1))
        features = features.reshape(batch_size, patch_count + 1, -1)

        global_features = features[:, 0]
        local_features = features[:, 1:]

        combined = torch.cat([
            global_features,
            local_features.mean(dim=1),
            local_features.std(dim=1, unbiased=False),
        ], dim=1)

        return combined

    def forward(self, whole, patches):
        return self.head(self.extract_features(whole, patches))