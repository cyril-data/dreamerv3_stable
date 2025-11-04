from __future__ import annotations
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch.nn as nn
import torch
import gymnasium as gym

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_device


class LSFMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        num_layers: int = 2,
        use_batchnorm: bool = True,
        activation_fn: nn.Module = nn.LeakyReLU(0.1),
    ) -> None:
        super().__init__(observation_space, features_dim)

        layers = [nn.Flatten()]
        input_dim = observation_space.shape[0]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, features_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(features_dim))
            layers.append(activation_fn)
            input_dim = features_dim  # Mise à jour pour la couche suivante

        self.linear = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
