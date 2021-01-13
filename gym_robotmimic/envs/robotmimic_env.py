import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
from gym_robotmimic.EnvRobot import EnvRobot

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
        
        self.robot.setGoalRobot(base_position=base_position,
                                init_links=init_links,
                                init_angles=init_angles)
                
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([35, 35]),
            dtype=np.uint)
        
        self.viewer = None
        self.cnt_step = 0
        
        #self.reset()
    
    def step(self, action):
        self.cnt_step += 1
        
        exit = self._applyAction(action)

        state = self._getState()
        reward = self._getReward(action)
        done = self.cnt_step >= 2000 or exit
        info = {}
        
        return (state, reward, done, info)
  
    def reset(self):
        val = self.robot.getRandomizeGoalJoints(-90, 90, 5)
        val = [None if i in range(0,1) else x for i, x in enumerate(val)]
        self.robot.setGoalJoints(val)

        self.robot.reset()
        self.cnt_step = 0
        return self._getState()
    
    def render(self, mode='human', close=False):
        self.robot.plot(goal=True)
        
    def render(self, mode='human'):
        img = np.array(self.robot.plot(goal=True))
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def goal(self, base_position, init_links, init_angles):
        self.robot.setGoalRobot(base_position, init_links, init_angles)

    def _getReward(self, action):
        return self.robot.getReward()
        
    def _applyAction(self, action):
        if action == 0:
            self._action_minus(self.robot, 1)
        elif action == 1:
            self._action_stay(self.robot, 1)
        elif action == 2:
            self._action_plus(self.robot, 1)
        else:
            raise AttributeError("Action not implemented: {}".format(action))
        
        return False
        
    def _getState(self):
                #i0 = int((robot_goal_state[1] + 90*math.pi/180) / (5*math.pi/180)) # goal robot state 1 
        #i1 = int((robot_goal_state[2] + 90*math.pi/180) / (5*math.pi/180)) # goal robot state 2
        #i2 = int((self.robot.links[1]['angle'] + 90*math.pi/180) / (5*math.pi/180)) # robot state 1 
        #s0 = int((self.robot.GoalAngles[1] + 90*math.pi/180) / (5*math.pi/180))
        
        s0 = int((self.robot.GoalAngles[1] + 90*math.pi/180) / (5*math.pi/180))
        s2 = int((self.robot.angles[1] + 90*math.pi/180) / (5*math.pi/180))
        return [s0, s2]

    def _action_plus(self, robot, link):
        if robot.links[link]['angle'] < 85*math.pi/180:
            a = [0, 0]
            a[link] = 5*math.pi/180
            robot.turn(a)
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_minus(self, robot, link):
        if robot.links[link]['angle'] > -85*math.pi/180:
            a = [0, 0]
            a[link] = -5*math.pi/180
            robot.turn(a)
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_stay(self, robot, link):
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    
    
class RobotMimicEnv_1(RobotMimicEnv):
    def __init__(self, base_position=(75, 75),
                       init_links=[(20, 10), (40, 5)],
                       init_angles=[0, 0],
                       img_w=150, img_h=150):
        RobotMimicEnv.__init__(self, base_position,
                       init_links,
                       init_angles,
                       img_w, img_h)
        self.action_space = spaces.Discrete(6)
        
    def reset(self):
        val = self.robot.getRandomizeGoalJoints(0, 360, 5)
        #val = [None if i in range(0,1) else x for i, x in enumerate(val)]
        self.robot.setGoalJoints(val)

        self.robot.reset()
        self.cnt_step = 0
        return self._getState()
    
    def _applyAction(self, action):
        if action == 0:
            self._action_minus(self.robot, 0)
        elif action == 1:
            self._action_minus(self.robot, 1)
        elif action == 2:
            self._action_stay(self.robot, 0)
        elif action == 3:
            self._action_stay(self.robot, 1)
        elif action == 4:
            self._action_plus(self.robot, 0)
        elif action == 5:
            self._action_plus(self.robot, 1)
        else:
            raise AttributeError("Action not implemented: {}".format(action))
        return False

    def _getReward(self, action):
        bonus_gain = 1
        if action == 2 or action == 3:
            bonus_gain = 1.1
        return self.robot.getReward() * bonus_gain
    
    def _action_plus(self, robot, link):
        a = [0, 0]
        a[link] = 5*math.pi/180
        robot.turn(a)
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_minus(self, robot, link):
        a = [0, 0]
        a[link] = -5*math.pi/180
        robot.turn(a)
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 

    def _action_stay(self, robot, link):
        return int((robot.links[link]['angle'] + 90*math.pi/180) / (5*math.pi/180)) 