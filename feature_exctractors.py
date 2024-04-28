# python utils
from datetime import datetime
import os
import random

# RL packages
import gymnasium

import minigrid
import numpy as np
import stable_baselines3
import torch
from torch import nn
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback


# Custom packages
from goal_env import SimpleEnv
from full_observable import OneHotFullyObsWrapper
from goal_conditioned_wrappers import GoalSpecifiedWrapper


class MinigridFeaturesExtractor(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.Space, features_dim: int = 512, normalized_image: bool = False, num_layer=3) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        
        
        
        
        
        if num_layer == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif num_layer == 5:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
            
        elif num_layer == 8:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
        else :
            raise('Invalid number of layers')

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class FeaturesStateGridGoalPos(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512, normalized_image: bool = False, num_layer=3) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['image'].shape[0]
        if num_layer == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
        else :
            raise('Invalid number of layers')

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1] #+ observation_space['goal'].shape[0]
        print(f"n_flatten {n_flatten}")
        print(f"embedding for image is {features_dim - observation_space['goal'].shape[0] } and {observation_space['goal'].shape[0]} for goal")
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim - observation_space['goal'].shape[0]), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations['image']
        goal = observations['goal']
        embed = self.linear(self.cnn(grid))
        # out = torch.zeros((embed.shape[0] + goal.shape[0]))
        return torch.cat((embed, goal), dim=1)
        # out[0:embed.shape[0]] = embed
        # out[embed.shape[0]:] = goal


class SameGoalStateEncoder(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512, normalized_image: bool = False, num_layer=3) -> None:
        super().__init__(observation_space, features_dim)
        assert observation_space['image'].shape == observation_space['goal'].shape
        n_input_channels = observation_space['image'].shape[0]
        if num_layer == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
        else :
            raise('Invalid number of layers')

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]

        assert features_dim % 2 == 0
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim // 2), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations['image']
        goal = observations['goal']
        state_embed = self.linear(self.cnn(grid))
        goal_embed = self.linear(self.cnn(grid))
        return torch.cat((state_embed, goal_embed), dim=1)
