from collections import OrderedDict

from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from gymnasium import logger, spaces
import numpy as np

from goal_env import SimpleEnv


class GoalSpecifiedWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(self.env.grid.height,
                   self.env.grid.height,
                   self.observation_space.spaces["image"].shape[2]*2)
        )
        # self.observation_space = spaces.Dict(
        #     {**self.observation_space.spaces, "image": env.observation_space.spaces["image"]}
        # )

    def observation(self, obs):        
        return np.concatenate([obs["image"], obs["goal"]], axis=2)



class GoalAndStateDictWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "image": self.observation_space.spaces['image'],
            "goal": self.observation_space.spaces['goal'],
        })
        print(self.observation_space)
        #     spaces.Box(
        #     low=0, 
        #     high=255,
        #     shape=(self.env.grid.height,
        #            self.env.grid.height,
        #            self.observation_space.spaces["image"].shape[2]*2)
        # )
        # self.observation_space = spaces.Dict(
        #     {**self.observation_space.spaces, "image": env.observation_space.spaces["image"]}
        # )

    def observation(self, obs):   
        out = {}
        out['image'] = obs["image"]
        out['goal'] = obs["goal"]
        return out

class HERWrapper(SimpleEnv):
    def __init__(self, **kwargs):
        assert "goal_encode_mode" in kwargs and kwargs["goal_encode_mode"] in ["grid", "position"]
        assert "image_encoding_mode" in kwargs and kwargs["image_encoding_mode"] in ["grid", "position"]
        assert kwargs["goal_encode_mode"] == kwargs["image_encoding_mode"]
        assert 'agent_in_goal' in kwargs and kwargs['agent_in_goal']
        super().__init__(**kwargs)
        assert self.observation_space.spaces['goal'] == self.observation_space.spaces['image']
        self.observation_space = spaces.Dict({
            "observation": self.observation_space.spaces['goal'],
            "achieved_goal": self.observation_space.spaces['goal'],
            "desired_goal": self.observation_space.spaces['goal'],
        })
        print(self.observation_space)

    def step(self, action):
        obs, reward, done, terminated, info = super().step(action)
        if reward > 0:
            reward = 1
        
        return OrderedDict (
            [
                ("observation", obs["image"]),
                ("achieved_goal", obs["image"]),
                ("desired_goal", obs["goal"]),
            ]
        ), reward, done, terminated, info
    
    def reset(self,**kwargs):
        obs, info = super().reset(**kwargs)
        return OrderedDict (
            [
                ("observation", obs["image"]),
                ("achieved_goal", obs["image"]),
                ("desired_goal", obs["goal"]),
            ]
        ), info

    def compute_reward(self, achieved_goal, desired_goal, _info) -> np.float32:
        # import ipdb; ipdb.set_trace()
        result = np.all(achieved_goal == desired_goal, axis=(1, 2, 3))
        return result.astype(int)
