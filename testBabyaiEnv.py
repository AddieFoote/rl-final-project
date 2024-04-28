# python utils
from datetime import datetime
import os
import random

# RL packages
import argparse
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
from goal_conditioned_wrappers import GoalSpecifiedWrapper, GoalAndStateDictWrapper
from feature_exctractors import MinigridFeaturesExtractor, FeaturesStateGridGoalPos, SameGoalStateEncoder


def train_with_PPO(model, log_dir, num_timesteps=2e5, eval_callback=None):
    if eval_callback is not None:
        model = model.learn(num_timesteps, callback = eval_callback) 
    else:
        model = model.learn(num_timesteps) # train
    model.save(log_dir)

def write_args_to_file(args, file_path):
    with open(file_path, "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

def make_env(args, rank):
    if args.env == 'custom-set-goal':
        env = SimpleEnv(render_mode="rgb_array", size = args.size, goal_pos = [2, 2], goal_encode_mode='position')
        # import ipdb; ipdb.set_trace()
        print_label_for_env = "custom_env"
    elif args.env == 'custom-dynamic' and args.obs == 'fully-observable':
        if args.goal_features == 'one-pos':
            goal_encode_mode = 'position'
        else:
            goal_encode_mode = 'grid'
        env = SimpleEnv(render_mode="rgb_array", goal_encode_mode=goal_encode_mode, image_encoding_mode='grid', size=args.size, reward_shaping=args.reward_shaping)
        if args.goal_features == "fully-observable":
            env = GoalSpecifiedWrapper(env)
        else:
            env = GoalAndStateDictWrapper(env)
        print_label_for_env = "custom_env"
    elif args.env == 'room':
        env = gymnasium.make("BabyAI-OneRoomS8-v0" , render_mode="rgb_array")
        print_label_for_env = "BabyAI-OneRoomS8-v0"


        
    if args.policy == 'CnnPolicy' and (args.algorithm == 'DQN' or args.algorithm == 'HER'):
        raise('DDPG and HER do not support CnnPolicy')
    if args.env == 'custom-dynamic' and args.obs == 'fully-observable':
        pass
    elif args.obs == "one-hot":
        env = minigrid.wrappers.OneHotPartialObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.obs == 'img':
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.obs == 'fully-observable':
        env = minigrid.wrappers.FullyObsWrapper(env)    
        env = minigrid.wrappers.ImgObsWrapper(env)
    elif args.obs == "fully-observable-one-hot":
        env = OneHotFullyObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
    else:
        raise('Invalid obs type selected')
    
    # if env.seed == 0:
    #     env.seed(random.randint(0, 50000))
    # else:
    #     env.seed(env.seed + rank)
        
    return env, print_label_for_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description here")
    parser.add_argument('--env', choices=['custom-set-goal', 'custom-dynamic', 'room'], default='custom-set-goal', help="Environment type")
    parser.add_argument('--obs', choices=['one-hot', 'img', "fully-observable", "fully-observable-one-hot"], default='img', help="Environment type (normalized or one-hot)")
    parser.add_argument('--algorithm', choices=['PPO', 'A2C', 'DDPG', 'DQN', 'HER', 'SAC', 'TD3'], default='PPO')
    parser.add_argument('--policy', choices=['CnnPolicy', 'MlpPolicy'], default='CnnPolicy')
    parser.add_argument('--save-id', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="save-id")
    parser.add_argument('--num-timesteps', type=int, default=2e5, help="Number of timesteps to train for")
    parser.add_argument('--num-conv-layers', type=int, default=3, help="Number of convolutional layers")
    parser.add_argument('--num_envs', type=int, default = 1, help="Number of environments to run in parallel - 1 means no parallelization")
    parser.add_argument('--size', type=int, default = 5, help="size of square of environment")
    parser.add_argument("--goal-features", type=str, choices=['fully-observable', 'one-pos', 'same-network-fully-obs'], default='fully-observable')
    parser.add_argument("--reward-shaping", type=bool, default=False)
    
    args = parser.parse_args()

    if args.size != 5: assert 'custom' in args.env
    if args.goal_features != "fully-observable": 
        assert args.env == 'custom-dynamic'
        args.policy = 'MultiInputPolicy'
        print("Using Multi Input Policy")

    
    #env, print_label_for_env = make_env(args.env, args)
    if args.goal_features == "fully-observable":
        features_extractor_class=MinigridFeaturesExtractor
    elif args.goal_features == "one-pos":
        features_extractor_class=FeaturesStateGridGoalPos 
    elif args.goal_features == 'same-network-fully-obs':
        features_extractor_class=SameGoalStateEncoder

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=dict(features_dim=128, num_layer=args.num_conv_layers),
    )

    eval_callback = None
    if args.num_envs > 1:
        print_label_for_env = make_env(args, 0)[1] + "_parallel"
        envfunc = lambda: make_env(args, 0)[0]
        env = DummyVecEnv([envfunc for i in range(4)])
        
        eval_env = envfunc()
        logs_path = os.path.join("./logs", print_label_for_env)
        log_dir = os.path.join(logs_path, args.algorithm, args.save_id)
        eval_callback = EvalCallback(eval_env, log_path=log_dir, best_model_save_path=log_dir, eval_freq= 2048, n_eval_episodes = 12, render = False)
    else:
        env, print_label_for_env = make_env(args, 0)
    
    
    policy_type = args.policy
    logs_path = os.path.join("./logs", print_label_for_env)
    log_dir = os.path.join(logs_path, args.algorithm, args.save_id)
    

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
        raise('HER not implemented')
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

    train_with_PPO(model, log_dir, num_timesteps=args.num_timesteps, eval_callback=eval_callback)