from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
from gymnasium import logger, spaces
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper


class OneHotFullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a one-hot encoding of the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.tile_size = 8
        width, height = self.env.width, self.env.height
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + len(DIR_TO_VEC)
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height, num_bits),
            dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        # Convert the full grid to a one-hot encoding
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
        for i in range(full_grid.shape[0]):
            for j in range(full_grid.shape[1]):
                type = full_grid[i, j, 0]
                color = full_grid[i, j, 1]
                state = full_grid[i, j, 2]
                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                if type == OBJECT_TO_IDX["agent"]:
                    out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + state] = 1
                else:
                    out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}