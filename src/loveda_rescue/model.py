"""ImageNet ResNet18 encoder and the first-round compact U-Net decoder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DecoderBlock(nn.Module):
    """Resize, concatenate skip, then two Conv3x3/BatchNorm/ReLU operations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class ResNet18UNet(nn.Module):
    """Load only the verified official local file; never trigger implicit downloads."""

    def __init__(self, weights_path: Path, config: dict[str, Any]) -> None:
        super().__init__()
        if config["weights_enum"] != "ResNet18_Weights.IMAGENET1K_V1":
            raise ValueError("Unsupported weight enum")
        transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
        if list(transforms.mean) != config["normalization_mean"] or list(transforms.std) != config["normalization_std"]:
            raise ValueError("Normalization differs from official weights")
        self.weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if not self.weights_sha256.startswith(config["weights_sha256_prefix"]):
            raise ValueError("Official weight hash prefix mismatch")
        encoder = resnet18(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)
        if not torch.equal(encoder.conv1.weight.detach(), state["conv1.weight"]):
            raise AssertionError("Encoder weights did not load")
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.decoder = nn.ModuleList([
            DecoderBlock(512 + 256, 128), DecoderBlock(128 + 128, 64),
            DecoderBlock(64 + 64, 32), DecoderBlock(32 + 64, 16)])
        self.head = nn.Conv2d(16, config["num_classes"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x = self.layer4(s3)
        for block, skip in zip(self.decoder, (s3, s2, s1, s0)):
            x = block(x, skip)
        return self.head(F.interpolate(x, size=size, mode="bilinear", align_corners=False))

    def describe(self) -> dict[str, Any]:
        """Record architectural and weight provenance information."""
        total = sum(p.numel() for p in self.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters()) + sum(p.numel() for p in self.head.parameters())
        return {"encoder": "torchvision ResNet18 IMAGENET1K_V1; fc discarded; all layers trainable",
                "decoder": "bilinear + skip + double Conv3x3/BN/ReLU; channels 128,64,32,16; 1x1 head -> 7",
                "parameters_total": total, "parameters_encoder": total - decoder,
                "parameters_decoder_and_head": decoder, "weights_sha256": self.weights_sha256}
