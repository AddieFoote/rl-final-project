import gym
from gymnasium import logger, spaces
import numpy as np
import torch
import minigrid.wrappers as wrappers
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Key
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Floor, Ball
# from minigrid.manual_control import ManualControl
from our_manual_control import OurManualControl as ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

import random


# the enivironment is zero indexed
# it also spawns walls around everything

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=None,
        agent_start_dir=0,
        goal_pos=None,
        max_steps: int | None = None,
        goal_encode_mode=None,
        image_encoding_mode='img', 
        number_of_balls=1,
        **kwargs,
    ):
        self.number_of_balls=number_of_balls
        assert self.number_of_balls < size**2 // 4
        
        self.goal_pos = goal_pos
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        self.goal_encode_mode = goal_encode_mode
        self.image_encoding_mode = image_encoding_mode
        if self.goal_pos == None: assert self.goal_encode_mode != None

        assert self.goal_encode_mode in [None, 'grid', 'position']
        assert self.image_encoding_mode in ['grid', 'img']
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2
        
        if self.goal_pos is not None: assert self.number_of_balls == 1


        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        if self.goal_encode_mode == 'grid':
            self.observation_space['goal'] = spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)
        elif self.goal_encode_mode == 'position':
            self.observation_space['goal'] = spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        
        if self.image_encoding_mode == 'grid':
            self.observation_space['image'] = spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)

        # self._gen_grid(size, size)


    @staticmethod
    def _gen_mission():
         
        return "grand mission"

    def place_our_obj(self, grid, obj, max_tries=100, forced_x = None, forced_y = None) -> tuple[int, int]:
        w, h = lambda : random.randint(0, grid.width - 1) , lambda : random.randint(0, grid.height - 1)
        for _ in range(max_tries):
            if forced_x is not None and forced_y is not None:
                #print("using forced position")
                pos = (forced_x, forced_y)
            elif forced_x is not None or forced_y is not None:
                raise RuntimeError("only passed one forced pos")
            else:
                pos = (w(), h())

                
            if grid.get(*pos) is None:
                grid.set(*pos, obj)

                obj.cur_pos = pos
                obj.init_pos = pos
                return pos            
        raise RuntimeError("bro how did we not get it")      

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height) # Create an empty grid
        self.goal_grid = Grid(width, height)
        
        self.grid.wall_rect(0, 0, width, height) # Generate the surrounding walls
        self.goal_grid.wall_rect(0, 0, width, height)

        # for i in range(0, height):  # Generate vertical separation wall
        #     self.grid.set(5, i, Wall())
       
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))  # Place the door and key
        self.real_balls = []
        self.goal_balls = []
        for i in range(self.number_of_balls):
            self.real_balls.append(Ball(COLOR_NAMES[0]))
            self.goal_balls.append(Ball(COLOR_NAMES[0]))

        for ball in self.real_balls:
            obj_pos = self.place_our_obj(self.grid, ball)

        # set_x_temp = 4
        # set_y_temp = 4
        if self.goal_pos is None:
            for ball in self.goal_balls:
                goal_obj_pos = self.place_our_obj(self.goal_grid, ball)
        else:
            goal_obj_pos = self.place_our_obj(self.goal_grid, self.goal_balls[0], forced_x = self.goal_pos[0], forced_y = self.goal_pos[1])

        #print(f"Our goal is located at {goal_obj_pos}")

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"
        self.set_the_goal()

    def check_goal_reached(self):
        if self.goal_encode_mode == "position":
            assert self.number_of_balls, "only one ball allowed for position encoding of goals"
            return (self.goal_balls[0].cur_pos[0] == self.real_balls[0].cur_pos[0]) and (self.goal_balls[0].cur_pos[1] == self.real_balls[0].cur_pos[1])
        elif self.goal_encode_mode == "grid":
            # find agent position
            
            # make a copy and remove the agent and replace it with nothing
            our_grid_copy = self.grid.encode()
            goal_grid_copy = self.goal_grid.encode()
            # compare two grids
            
            return (our_grid_copy == goal_grid_copy).all()
        else:
            raise ValueError("wrong input to check_goal_reached lol")
    
            
            
    def step(self, action):
        obs, reward, done, terminated, info = super().step(action)
        if self.image_encoding_mode == 'grid':
            full_grid = self.grid.encode()
            full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array(
                [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir]
            )
            obs['image'] = full_grid
        if self.goal_encode_mode != None:
            obs['goal'] = self.goal_encoded
        reward = 0
        
        if self.check_goal_reached():
            reward = self._reward()
            # if reward != 0:
            #     print('got the reward')
            terminated = True
            done = True
        return obs, reward, done, terminated, info

    def reset(self, **kwargs):
        # don't spawn with the goal already reached
        reset_once = False
        while not reset_once or self.check_goal_reached(): 
            obs, info = super().reset(**kwargs)
            if self.image_encoding_mode == 'grid':
                full_grid = self.grid.encode()
                full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array(
                    [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir]
                )
                obs['image'] = full_grid
            if self.goal_encode_mode != None:
                obs['goal'] = self.goal_encoded
                
            reset_once = True
            # if self.check_goal_reached():
            #     print("resetting double")
        
        return obs, info

    def set_the_goal(self):
        if self.goal_encode_mode == "position":
            assert self.number_of_balls == 1, "only one ball currently allowed for position encoding of goals"
            self.goal_encoded = np.array(self.goal_balls[0].cur_pos)
        elif self.goal_encode_mode == "grid":
            # import ipdb; ipdb.set_trace()
            # self.observation_space.spaces["grid"]
            self.goal_encoded = self.goal_grid.encode()
        elif self.goal_encode_mode == None:
            self.goal_encoded = None
        else:
            raise ValueError("wrong input to get_the_goal lol")

def main():
    env = env = SimpleEnv(render_mode="human", goal_encode_mode='grid', image_encoding_mode='grid', size=6)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()