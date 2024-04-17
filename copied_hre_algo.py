import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium
import typing
import datetime
import os

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Intel MKL WARNING.*')


import os
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX'


Experience = namedtuple("Experience", field_names="state action reward next_state done")


class ImageZerothIndexWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        image_space = self.env.observation_space['image']
        image_shape = image_space.shape[:2]  # Get the height and width of the image
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(image_shape[0] * image_shape[1] + NUM_DIRECTIONS,),
            dtype=np.uint8)

    def observation(self, observation):
        full_obs = observation
    
        direction = full_obs['direction'].item()
        image = full_obs['image']
        
        if not isinstance(direction, int):
            print(direction)
            print(type(image))
            raise ValueError("direction is not int")
        if not isinstance(image, np.ndarray):
            print(image)
            print(type(image))
            raise ValueError("Image is not a numpy array")
        
        # Flatten the image and concatenate with direction one-hot encoding
        ar = np.zeros(image.shape[0] * image.shape[1] + NUM_DIRECTIONS, dtype=np.uint8)
        ar[:image.shape[0] * image.shape[1]] = image[:, :, 0].flatten()
        ar[image.shape[0] * image.shape[1] + direction] = 1
        
        return torch.tensor(ar).to(torch.float32)
    
    
class GoalAndState(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(2 + NUM_DIRECTIONS,),
            dtype=np.uint8)

    def observation(self, observation): 
        full_obs = observation
    
        direction = full_obs['direction'].item()
        image = full_obs['image']
        
        if not isinstance(direction, int):
            print(direction)
            print(type(image))
            raise ValueError("direction is not int")
        if not isinstance(image, np.ndarray):
            print(image)
            print(type(image))
            raise ValueError("Image is not a numpy array")
        
        position = np.where(image[:, :, 0] == 6)

        # Flatten the image and concatenate with direction one-hot encoding
        
        ar = np.zeros(2 + NUM_DIRECTIONS, dtype=np.uint8)
        # import ipdb; ipdb.set_trace()
        if position[0].shape == (0,):
            ar[0] = -1
            ar[1] = -1
        else:
            ar[0] = position[0][0]
            ar[1] = position[1][0]
            ar[2 + direction] = 1
        
        return torch.tensor(ar).to(torch.float32)
    

class DuelingMLP(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.linears = []
        self.CNNs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),  # Adjusted input size
            *[item for sublist in [[nn.ReLU(), nn.Linear(64, 64)] for _ in range(3)] for item in sublist],
            nn.ReLU(),
        )
        
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, num_actions)

    def forward(self, x):
        x = x.unsqueeze(0) if len(x.size()) == 1 else x
        
        map_input = x[:, :49].reshape((-1, 7, 7))
        map_output = self.CNNs(map_input.unsqueeze(1))
        map_output = map_output.squeeze(1).reshape((-1, 7*7))
        
        x = torch.cat((map_output, x[:, 49:]), dim=1)  # Concatenate map output with remaining features
        
        x = self.model(x)
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        action_values = (value + (advantage - advantage.mean(dim=1, keepdim=True))).squeeze()
        return action_values

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    """ Double-DQN with Dueling Architecture """

    def __init__(self, state_size, num_actions, path=None):

        self.state_size = state_size
        self.num_actions = num_actions

        self.gamma = 0.98
        self.batch_size = 128
        self.train_start = 1000

        self.memory = ReplayMemory(int(1e6))
        
        if path==None:
            self.Q_network = DuelingMLP(state_size, num_actions)
            self.target_network = DuelingMLP(state_size, num_actions)
        else:
            self.load_model(path, state_size, num_actions)
            
        self.update_target_network()

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=0.001)

    def push_experience(self, state, action, reward, next_state, done):
        self.memory.push(Experience(state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def take_action(self, state, epsilon):
        if random.random() > epsilon:
            return self.greedy_action(state)
        else:
            return torch.randint(self.num_actions, size=())

    def greedy_action(self, state):
        with torch.no_grad():
            return self.Q_network(state).argmax()
    
    def save_model(self, path="."):
        torch.save(self.target_network.state_dict(), os.join(path, datetime.now(), "/target"))
        torch.save(self.Q_network.state_dict(), os.join(path, datetime.now(), "/Q_network"))
    
    def load_model(self, path, state_size, num_actions):
        self.target_network = DuelingMLP(state_size, num_actions)
        self.target_network.load_state_dict(torch.load(os.join(path, "/target")))
        self.Q_network = DuelingMLP(state_size, num_actions)
        self.Q_network.load_state_dict(torch.load(os.join(path, "/Q_network")))

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])

        Q_values = self.Q_network(state_batch)[range(self.batch_size), action_batch]

        # Double DQN target #
        next_state_values = torch.zeros(self.batch_size)
        number_of_non_final = sum(non_final_mask)
        with torch.no_grad():
            argmax_actions = self.Q_network(non_final_next_states).argmax(1)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
                range(number_of_non_final), argmax_actions]

        Q_targets = reward_batch + self.gamma * next_state_values
        ####################
        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        self.optimizer.step()



SOMETHING_IDK_WHAT = 200
NUM_DIRECTIONS = 4
EPISODES_PER_BATCH = 16

def true_print(item):
    with open("true_output.txt", "a") as file:
        file.write(item + "\n")

def reset_true_output():
    with open("true_output.txt", "w") as file:
        pass


def generate_env(human = False, wrapper=ImageZerothIndexWrapper):
    if human:
        env: gymnasium.Env = gymnasium.make("BabyAI-OneRoomS8-v0", max_steps = SOMETHING_IDK_WHAT, render_mode="human") # TODO: change documentation
    else:
        env: gymnasium.Env = gymnasium.make("BabyAI-OneRoomS8-v0", max_steps = SOMETHING_IDK_WHAT)#, render_mode="human") # TODO: change documentation
    env = wrapper(env)
    return env
    


def train(env, num_actions, state_size, num_epochs=10, hindsight_replay=True,
          eps_max=0.2, eps_min=0.0, exploration_fraction=0.5):

    """
    Training loop using DQN or DQN with
    hindsight experience replay. Exploration is decayed linearly from eps_max to eps_min over a fraction of the total
    number of epochs according to the parameter exploration_fraction. Returns a list of the success rates over the
    epochs.
    """

    # Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
    future_k = 4
    num_cycles = 50
    num_episodes = EPISODES_PER_BATCH
    num_opt_steps = 40

    agent = DQNAgent(state_size, num_actions)

    success_rate = 0.0
    success_rates = []
    for epoch in range(num_epochs):
        
        # Decay epsilon linearly from eps_max to eps_min
        eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)
        
        
        true_print("Epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * eps, success_rate))

        successes = 0
        for cycle in range(num_cycles):
            for episode in range(num_episodes):
                
                if episode == 0 and cycle == 0:
                    env = generate_env(human = True)
                elif episode == 1 and cycle == 0:
                    env = generate_env(human = False)
                    true_print("Generated new env")

                # Run episode and cache trajectory
                episode_trajectory = []
                
                
                state, info = env.reset()
                goal = torch.tensor([-1.])

                for step in range(SOMETHING_IDK_WHAT):
                    state_ = torch.cat((state, goal))
                    action = agent.take_action(state_, eps)
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    # if terminated:
                    #     true_print("Terminated")
                        
                    # if truncated:
                    #     true_print("Truncated")
                    
                    reward = torch.tensor(reward)
                    done = terminated or truncated
                    episode_trajectory.append(Experience(state, action, reward, next_state, done))
                    state = next_state
                    if terminated:
                        successes += 1
                    if truncated:
                        assert step == SOMETHING_IDK_WHAT - 1
                    if done:
                        break

                # Fill up replay memory
                steps_taken = step
                for t in range(steps_taken):

                    # Standard experience replay
                    state, action, reward, next_state, done = episode_trajectory[t]
                    state_, next_state_ = torch.cat((state, goal)), torch.cat((next_state, goal))
                    agent.push_experience(state_, action, reward, next_state_, done)

                    # Hindsight experience replay
                    if hindsight_replay:
                        for _ in range(future_k):
                            future = random.randint(t, steps_taken)  # index of future time step
                            new_goal = episode_trajectory[future].next_state  # take future next_state and set as goal
                            new_reward, new_done = env.compute_reward(next_state, new_goal)
                            state_, next_state_ = torch.cat((state, new_goal)), torch.cat((next_state, new_goal))
                            agent.push_experience(state_, action, new_reward, next_state_, new_done)

            # Optimize DQN
            for opt_step in range(num_opt_steps):
                agent.optimize_model()

            agent.update_target_network()

        success_rate = successes / (num_episodes * num_cycles)
        success_rates.append(success_rate)

    return success_rates


if __name__ == "__main__":
    reset_true_output()
    epochs = 40

    for her in [False]: # True
        
        env = generate_env(human = False)#), wrapper=GoalAndState)
        # env = GoalAndState(env)
        obs, info = env.reset() # TODO: make an issue on this this is insane does nobody use this env
        
        #import ipdb; ipdb.set_trace()
        success = train(env, env.action_space.n, env.observation_space.shape[0] + 1, epochs, her, eps_max=0.3)
        
        
        plt.plot(success, label="HER-DQN" if her else "DQN")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.show()
