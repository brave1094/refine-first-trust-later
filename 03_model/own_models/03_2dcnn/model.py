"""
load_model_cnn.py
─────────────────
2D-CNN (LeNet-5 style)
Paper: Wang et al., ICOIN 2017

Architecture:
  Input  : (B, 1, 28, 28)
  C1     : Conv2d(1,  32, 5×5, padding=2) → (B, 32, 28, 28)  ReLU
  P1     : MaxPool2d(2×2)                 → (B, 32, 14, 14)
  C2     : Conv2d(32, 64, 5×5, padding=2) → (B, 64, 14, 14)  ReLU
  P2     : MaxPool2d(2×2)                 → (B, 64,  7,  7)
  Flatten: 64×7×7 = 3136
  FC1    : 3136 → 1024  ReLU  Dropout(0.5)
  FC2    : 1024 → num_classes
"""

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # P1
            nn.MaxPool2d(kernel_size=2, stride=2),
            # C2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # P2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 28, 28) or (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)          # → (B, 1, 28, 28)
        x = self.features(x)
        x = self.classifier(x)
        return x                         # logits (B, num_classes)