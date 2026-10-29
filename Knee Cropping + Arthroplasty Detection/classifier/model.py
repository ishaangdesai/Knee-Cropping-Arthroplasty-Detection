"""Model architecture for the knee metal/regular classifier.

EfficientNet-B1 transfer learning (weights already cached locally on this
machine from the group's Faster R-CNN project, so this trains without
needing a fresh internet download). Head is replaced with a single logit -
sigmoid(logit) is the probability the knee has a metal replacement.

FREEZE_BACKBONE was True (only the last 2 feature blocks trainable) back
when the labeled set was 174 images / 21 positive - full fine-tuning would
have overfit almost immediately at that size. The labeled set is now 1732
images / 194 positive, which is enough to fine-tune the whole network, so
this defaults to False. If training curves show val loss diverging from
train loss early (classic overfitting), that's the first knob to revisit.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

ARCHITECTURE = "efficientnet_b1"
FREEZE_BACKBONE = False
UNFROZEN_FEATURE_BLOCKS = 2  # only used if FREEZE_BACKBONE=True: trailing entries in model.features left trainable


def build_model(pretrained: bool = True, freeze_backbone: bool = FREEZE_BACKBONE) -> nn.Module:
    weights = EfficientNet_B1_Weights.IMAGENET1K_V2 if pretrained else None
    model = efficientnet_b1(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        for block in list(model.features)[-UNFROZEN_FEATURE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True

    return model
