# python utils
from datetime import datetime
import os

# RL packages
import argparse
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

def train_with_PPO(model, log_dir, num_timesteps=2e5):
    model = model.learn(num_timesteps) # train for 200k timesteps
    model.save(log_dir)

def write_args_to_file(args, file_path):
    with open(file_path, "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description here")
    parser.add_argument('--env', choices=['one-hot', 'img', "fully-observable", "fully-observable-one-hot"], default='img', help="Environment type (normalized or one-hot)")
    parser.add_argument('--algorithm', choices=['PPO', 'A2C', 'DDPG', 'DQN', 'HER', 'SAC', 'TD3'], default='PPO')
    parser.add_argument('--policy', choices=['CnnPolicy', 'MlpPolicy'], default='CnnPolicy')
    parser.add_argument('--save-id', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="save-id")
    args = parser.parse_args()

    env = gymnasium.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
    if args.policy == 'CnnPolicy' and (args.algorithm == 'DQN' or args.algorithm == 'HER'):
        raise('DDPG and HER do not support CnnPolicy')
    
    if args.env == "one-hot":
        env = minigrid.wrappers.OneHotPartialObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.env == 'img':
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.env == 'fully-observable':
        env = minigrid.wrappers.FullyObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.env == "fully-observable-one-hot":
        raise('Fully observable one-hot not implemented yet')
        # TODO: Implement fully observable one-hot
    else:
        raise('Invalid env selected')
    
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    policy_type = args.policy
    log_dir = os.path.join("./logs", args.algorithm, args.save_id)

    if args.algorithm == 'PPO':
        model = stable_baselines3.PPO(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'A2C':
        model = stable_baselines3.A2C(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'DDPG':
        raise('DDPG not applicable, expects continuous action space')
        model = stable_baselines3.DDPG(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'DQN':
        model = stable_baselines3.DQN(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'HER':
        raise('HER not implemented yet')
        # TODO: Implement HER, use fully observable so we can use states as goals
        model = stable_baselines3.HER(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'SAC':
        raise('SAC not applicable, expects continuous action space')
        model = stable_baselines3.SAC(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    elif args.algorithm == 'TD3':
        raise('TD3 not applicable, expects continuous action space')
        model = stable_baselines3.TD3(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, verbose=1)
    else:
        raise('Invalid algorithm selected')
    
    os.makedirs(log_dir, exist_ok=True)
    write_args_to_file(args, log_dir + "/arguments.txt")

    train_with_PPO(model, log_dir)