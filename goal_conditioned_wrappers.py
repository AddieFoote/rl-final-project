from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from gymnasium import logger, spaces
import numpy as np

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

