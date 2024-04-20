from collections import namedtuple
from functools import partial

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns; sns.set()

from constants import NUM_DIRECTIONS, NUM_STATE_CLASSES

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
    

class NormalizedImage(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        image_space = self.env.observation_space['image']
        image_shape = image_space.shape[:2]  # Get the height and width of the image
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
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
        ar = ar.astype(np.float32)
        ar /= 255.
        return torch.tensor(ar).to(torch.float32)

class OneHotImageZerothIndexWrapper(gymnasium.ObservationWrapper):
    """
    literally the above class but the image is instead actually one-hot encoded
    """
    def __init__(self, env, num_cell_states = NUM_STATE_CLASSES):
        super().__init__(env)
        image_space = self.env.observation_space['image']
        image_shape = image_space.shape[:2]  # Get the height and width of the image
        self.num_cell_states = num_cell_states
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(image_shape[0] * image_shape[1] * num_cell_states + NUM_DIRECTIONS,), dtype=np.float32)

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

        # One-hot encode the image based on the number of possible cell states
        image_tensor = torch.from_numpy(image[:, :, 0]).long()
        one_hot_image = torch.nn.functional.one_hot(image_tensor, num_classes=self.num_cell_states).float()
        one_hot_image = one_hot_image.permute(2, 0, 1).flatten() # put the class at the front and then flatten


        # Flatten the one-hot encoded image and concatenate with direction one-hot encoding
        ar = np.zeros(image.shape[0] * image.shape[1] * self.num_cell_states + NUM_DIRECTIONS, dtype=np.float32)
        ar[:image.shape[0] * image.shape[1] * self.num_cell_states] = one_hot_image
        ar[image.shape[0] * image.shape[1] * self.num_cell_states + direction] = 1

        
        return torch.tensor(ar)
    
    
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
    