import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from EnvRobot import EnvRobot

class RobotMimicEnv(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self, base_position=(0, 50),
                       init_links=[(20, 10), (40, 5)],
                       init_angles=[0, 0],
                       img_w=70, img_h=100):
        
        self.robot = EnvRobot(base_position, 
                              init_links, # [(arm length, width)]
                              init_angles) # [initial angles]
        
        self.robot.setObservationSpace(img_w, img_h)
        self.robot.setGoalRobot(base_position=(0, 50),
                                init_links=[(10, 10), (50, 7)],
                                init_angles=[0, 0])
                
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([38, 38]), dtype=np.uint)
        
        self.reset()
    
    def step(self, action):
        if action == 0:
            s2 = _action_plus(self.robot)
        elif action == 1:
            s2 = _action_minus(self.robot)
        elif action == 2:
            s2 = _action_stay(self.robot)
        else:
            raise("Action not implemented")

        #i0 = int((robot_goal_state[1] + 90*math.pi/180) / (5*math.pi/180)) # goal robot state 1 
        #i1 = int((robot_goal_state[2] + 90*math.pi/180) / (5*math.pi/180)) # goal robot state 2
        #i2 = int((self.robot.links[1]['angle'] + 90*math.pi/180) / (5*math.pi/180)) # robot state 1 
        s0 = int((self.robot.GoalAngles[1] + 90*math.pi/180) / (5*math.pi/180))
        
        state = [s0, s2]
        reward = self.robot.getReward()
        done = False
        info = {}
        
        return (state, reward, done, info)
  
    def reset(self):
        self.robot.reset()
    
    def render(self, mode='human', close=False):
        pass
    
    def goal(self, base_position, init_links, init_angles):
        self.robot.setGoalRobot(base_position, init_links, init_angles)


    def _action_plus(robot):
        if robot.links[1]['angle'] < 90*math.pi/180:
            robot.turn([0, 5*math.pi/180])
        return int((robot.links[1]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_minus(robot):
        if robot.links[1]['angle'] > -90*math.pi/180:
            robot.turn([0, -5*math.pi/180])
        return int((robot.links[1]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_stay(robot):
        return int((robot.links[1]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    
