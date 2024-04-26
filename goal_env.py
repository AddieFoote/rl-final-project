import gym
from gym import spaces
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

import random

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=None,
        agent_start_dir=0,
        goal_pos = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.goal_pos = goal_pos
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def place_our_obj(self, grid, obj, max_tries=100, forced_x = None, forced_y = None) -> tuple[int, int]:
        w, h = lambda : random.randint(0, grid.width - 1) , lambda : random.randint(0, grid.height - 1)
        for _ in range(max_tries):
            if forced_x is not None and forced_y is not None:
                print("using forced position")
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

        self.real_ball = Ball(COLOR_NAMES[0])
        self.goal_ball = Ball(COLOR_NAMES[0])

        obj_pos = self.place_our_obj(self.grid, self.real_ball)

        # set_x_temp = 4
        # set_y_temp = 4
        if self.goal_pos is None:
            goal_obj_pos = self.place_our_obj(self.goal_grid, self.goal_ball)
        else:
            goal_obj_pos = self.place_our_obj(self.goal_grid, self.goal_ball, forced_x = self.goal_pos[0], forced_y = self.goal_pos[1])

        print(f"Our goal is located at {goal_obj_pos}")

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def step(self, action):
        obs, reward, done, terminated, info = super().step(action)
        reward = 0
        
        
        if (self.goal_ball.cur_pos[0] == self.real_ball.cur_pos[0]) and (self.goal_ball.cur_pos[1] == self.real_ball.cur_pos[1]):
            reward = self._reward()
            # if reward != 0:
            #     print('got the reward')
            terminated = True
            done = True
        return obs, reward, done, terminated, info


    def get_the_goal(self, mode="position"):
        if mode == "position":
            return self.goal_ball.cur_pos
        elif mode == "grid":
            return self.goal_grid.encode()
        else:
            raise ValueError("wrong input to get_the_goal lol")

def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()