# Goal-Conditioned Reinforcement Learning and Representations

[![Report](https://img.shields.io/badge/Technical%20Report-PDF-blue.svg)](https://drive.google.com/file/d/1n3ooE5cu6S69NMDq4pYM8buZOF6ciA0b/view?usp=sharing)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Official implementation of **"Goal-Conditioned Reinforcement Learning and Representations"** - an empirical study of goal-conditioned RL algorithms across different observation spaces and architectural configurations in procedurally generated environments. For detailed results and analysis, see our [technical report](https://drive.google.com/file/d/1n3ooE5cu6S69NMDq4pYM8buZOF6ciA0b/view?usp=sharing) and [presentation](https://www.youtube.com/watch?v=zSou_YvQP-Y).

## Abstract

Abstract—Goal-Conditioned Reinforcement Learning enables agents to learn policies that can accomplish a variety of goals. We investigate goal-conditioned Reinforcement learning in a custom grid world environment based on the BabyAI framework that enables us to specify the goal as a desired world state. We compare the performance of different RL algorithms, including PPO, A2C, and DQN. We also explore the impact of various state and goal representations along with network architectures for our function extractors. Our experiments show that PPO outperforms the other algorithms in our setup, and that concatenating fully observable state representations with goal states is an effective input representation for the network. To address challenges with sparse rewards in larger environments, we implement reward shaping based on the distance between the ball and the goal, which enables learning in 6x6 grid worlds. We also test hindsight experience replay, but find that it does not yield significant benefits and substantially underperforms PPO in our specific setup. Our findings demonstrate the potential of goal-conditioned RL for flexibly solving tasks with multiple goals and highlight the importance of appropriate state and goal representations. 

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/goal-conditioned-rl.git
cd goal-conditioned-rl

# Create environment
conda create -n gcrl python=3.12
conda activate gcrl

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Train a PPO agent on our custom dynamic environment:

```bash
python testBabyaiEnv.py \
    --env custom-dynamic \
    --obs fully-observable \
    --num-conv-layers 5 \
    --num_env 32 \
    --num-timesteps 4000000 \
    --size 5 \
    --reward-shaping True
```

## Experimental Configuration

### Core Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--env` | `custom-set-goal`, `custom-dynamic`, `room` | Environment variant |
| `--obs` | `one-hot`, `img`, `fully-observable`, `fully-observable-one-hot` | Observation representation |
| `--algorithm` | `PPO`, `A2C`, `DQN`, `HER` | RL algorithm |
| `--policy` | `CnnPolicy`, `MlpPolicy` | Policy network architecture |
| `--num-timesteps` | Integer | Total training timesteps |
| `--num-conv-layers` | `3`, `5`, `8` | CNN feature extractor depth |
| `--num_envs` | Integer | Number of parallel environments |
| `--size` | Integer | Environment grid size (square) |
| `--goal-features` | `fully-observable`, `one-pos`, `same-network-fully-obs`, `HER` | Goal representation method |
| `--reward-shaping` | `True`/`False` | Enable dense reward signals (default: False) |

We evaluate CNN architectures with 3, 5, and 8 convolutional layers to understand the impact of representation capacity on goal-conditioned learning efficiency.

#### Algorithm-Specific Notes

- **HER (Hindsight Experience Replay)**: Only compatible with `custom-dynamic` environment
- **Reward Shaping**: Optional dense reward signal for accelerated learning

## Monitoring and Visualization

Generate clean experiment logs:
```bash
make board env=target_env
```

Launch TensorBoard for result visualization:
```bash
tensorboard --logdir ./logs/clean/target_env
```
