# rl-final-project

This repository contains code for training and evaluating reinforcement learning (RL) agents in BabyAI environments using different algorithms and configurations.

## Installation

Clone the repository:
`git clone https://github.com/your-username/rl-project.git`

Install the required dependencies:
`pip install -r requirements.txt`


## Usage
To train an RL agent, you can run the main script with the desired command-line arguments. For example:
```python testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 4000000 --size 5 --reward-shaping True```

This command will train an agent using the PPO algorithm on our custom 5x5 Goal Conditioned environment with a feature extractor of 5 conv layers. It will use the fully-observable observation and use reward shaping.

You can customize the training process by modifying the command-line arguments. Here are some of the available options:
- `--env`: Environment type (e.g., `custom-set-goal`, `custom-dynamic`, `room`)
- `--obs`: Observation type (e.g., `one-hot`, `img`, `fully-observable`, `fully-observable-one-hot`)
- `--algorithm`: RL algorithm (e.g., `PPO`, `A2C`, `DQN`)
- `--policy`: Policy type (e.g., `CnnPolicy`, `MlpPolicy`)
- `--num-timesteps`: Number of timesteps to train for
- `--num-conv-layers`: Number of convolutional layers in the policy (3, 5, or 8)
- `--num_envs`: Number of parallel environments to run
- `--size`: Size of the square environment
- `--goal-features`: Goal feature representation (e.g., `fully-observable`, `one-pos`, `same-network-fully-obs`, `HER`)
- `--reward-shaping`: Enable reward shaping (default: False, don't include the flag if you don't want reward shaping)

## Visualizing Results
do `make board env=target_env` to create good logs of for the base `target_env` and to start the associated tensorboard
