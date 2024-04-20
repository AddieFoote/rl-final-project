# python utils
import datetime
import os

# RL packages
import gymnasium
import minigrid
import numpy as np
import stable_baselines3
import torch
from torch import nn


class MinigridFeaturesExtractor(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def train_with_PPO():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gymnasium.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
    env = minigrid.wrappers.ImgObsWrapper(env)

    model = stable_baselines3.PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(2e5)

if __name__ == "__main__":
    train_with_PPO()