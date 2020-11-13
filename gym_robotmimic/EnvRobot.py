#!/usr/bin/env python
# coding: utf-8

import math
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import numpy as np
import itertools
import random

class EnvRobot:
    #############################
    # OOP functions             #
    #############################
    def __init__(self, base_position, init_links, init_angles):
        self.setRobot(base_position, init_links, init_angles)
        self.setObservationSpace()
        self.mini = 0
        self.maxi = np.sum(self.getRobotState(self.links))

    def setObservationSpace(self, img_w=400, img_h=400):
        self.img_w = img_w
        self.img_h = img_h

    def setRobot(self, base_position, init_links, init_angles):
        links = [self.link(init_links[0][0], init_links[0][1], 0, base_position, 0)]
        for (l, b) in init_links[1:]:
            links.append(self.link(l, b))

        self.links = self.set_joints(links, init_angles)
        self.init_angles = init_angles
        self.angles = init_angles

    def setGoalRobot(self, base_position, init_links, init_angles):
        links = [self.link(init_links[0][0], init_links[0][1], 0, base_position, 0)]
        for (l, b) in init_links[1:]:
            links.append(self.link(l, b))

        self.GoalLinks = self.set_joints(links, init_angles)
        self.GoalAngles = init_angles


    def plotRobot(self, img, links, colors=["red", "green", "yellow"]):
        img1 = ImageDraw.Draw(img)
        self.plot_robot(links, img1, colors)
        return img1

    def plot(self, goal=False):
        img = Image.new("RGB", (self.img_w, self.img_h))

        if goal:
            self.plotRobot(img, self.GoalLinks, colors=["blue", "blue", "blue"])
        self.plotRobot(img, self.links)

        #imshow(np.asarray(img))
        return img


    def reset(self):
        self.links = self.set_joints(self.links, self.init_angles)
        self.angles = self.init_angles

    def turn(self, angles):
        #print(angles)
        self.links = self.add_to_joints(self.links, angles)
        self.angles = np.add(self.angles, angles).tolist()

    def getRobotState(self, links):
        img = Image.new("RGB", (self.img_w, self.img_h))
        self.plotRobot(img, links, colors=["white", "white", "white"])

        # Descretize image
        m = np.asarray(img)
        # sum all RGB and threshold with > 0
        m2 = np.where(np.sum(m, 2) > 0, 1, 0)
        return m2

    def getState(self):
        return self.getRobotState(self.links)

    def getReward(self):
        m_state = self.getRobotState(self.links)
        m_goal = self.getRobotState(self.GoalLinks)

        m = m_state * m_goal
        sum_px = np.sum(m)
        if self.maxi is not None:
            return (float)(sum_px - self.mini)/self.maxi
        else:
            return sum_px

    def getRewardScaling(self, start, stop, step):
        start_angles = [l['angle'] for l in self.links]
        print(start_angles)

        s = [np.arange(start*math.pi/180, stop*math.pi/180, step*math.pi/180) for _ in self.links]
        rewards = []
        for j in itertools.product(*s):
            self.turn(j)
            rewards.append(self.getReward())

        self.turn(start_angles)

        self.mini = min(rewards)
        self.maxi = max(rewards)
        return (self.mini, self.maxi)

    def getRandomizeGoalJoints(self, min_angle, max_angle, step=1):
        random_joints = [random.randrange(min_angle, max_angle, step)*math.pi/180.0 for _ in self.GoalLinks]
        return random_joints
    
    def setGoalJoints(self, joint_values):
        # Replace all values which are None with the current angles
        for i, val in enumerate(joint_values):
            if val is None:
                joint_values[i] = self.GoalAngles[i]
        self.GoalLinks = self.set_joints(self.GoalLinks, joint_values)
        self.GoalAngles = joint_values
        
    #############################
    # More functional functions #
    #############################

    # Link
    def link(self, length, width, angle=0, position=(0,0), orientation=0):
        w_2 = width/2;
        l_2 = length/2;
        px, py = position
        c, s = math.cos(orientation+angle), math.sin(orientation+angle)

        shape =  [(px, py-w_2),
                  (px+length, py-w_2),
                  (px+length, py+w_2),
                  (px, py+w_2)]
        #print(shape)
        rot_shape = [(c*(x-px) - s*(y-py) + px,
                      s*(x-px) + c*(y-py) + py) for (x,y) in shape]

        return {"shape":rot_shape,
                "center":position,
                "orientation":orientation,
                "length":length,
                "width":width,
                "angle":angle,
                "next_joint":(c*length+px, s*length+py),
                "next_orientation":angle+orientation}

    #shape2 = [(0, 80), (10, 300)]

    def plot_link(self, link, img1, colors=["red", "green", "yellow"]):
        img1.polygon(link["shape"], fill = colors[0])

        img1.ellipse([tuple(np.subtract(link["center"], (link["width"]/2, link["width"]/2))),
                 tuple(np.add(link["center"], (link["width"]/2, link["width"]/2)))], fill = colors[1])
        img1.ellipse([tuple(np.subtract(link["next_joint"], (link["width"]/2, link["width"]/2))),
             tuple(np.add(link["next_joint"], (link["width"]/2, link["width"]/2)))], fill = colors[2])
        #print link["center"]
        return img1

    def plot_robot(self, links, img, colors=["red", "green", "yellow"]):
        for link in links:
            self.plot_link(link, img, colors)
        return img

    def set_joints(self, links, angles):
        updated_links = links
        updated_links[0] = self.link(updated_links[0]["length"],
                                     updated_links[0]["width"],
                                     angles[0],
                                     updated_links[0]["center"],
                                     updated_links[0]["orientation"])
        for i, (l, angle) in enumerate(zip(updated_links[1:], angles[1:]),1):
            updated_links[i] = self.link(updated_links[i]["length"],
                                         updated_links[i]["width"],
                                         angles[i],
                                         updated_links[i-1]["next_joint"],
                                         updated_links[i-1]["next_orientation"])
        return updated_links

    def add_to_joints(self, links, angles):
        updated_links = links
        updated_links[0] = self.link(updated_links[0]["length"],
                                     updated_links[0]["width"],
                                     updated_links[0]["angle"] + angles[0],
                                     updated_links[0]["center"],
                                     updated_links[0]["orientation"])

        for i, (l, angle) in enumerate(zip(updated_links[1:], angles[1:]),1):
            updated_links[i] = self.link(updated_links[i]["length"],
                                         updated_links[i]["width"],
                                         updated_links[i]["angle"] + angles[i],
                                         updated_links[i-1]["next_joint"],
                                         updated_links[i-1]["next_orientation"])
        return updated_links

if __name__ == "__main__":
    pos = (0, 200)
    env = EnvRobot(pos,
           [(100, 20), (100, 15), (100, 10), (100, 5)], # arm length and width
           [0, 0, 0, 0]) # initial angles

    env.setObservationSpace(400, 400)

    env.setGoalRobot(pos,
             [(100, 20), (100, 15), (100, 10), (100, 5)], # arm length and width
             [0, -45*math.pi/180, -90*math.pi/180, -90*math.pi/180]) # initial angles

    env.turn([0, 45*math.pi/180, 90*math.pi/180, 90*math.pi/180])
    env.plot(goal=True)

    m = env.getState()
    print(env.getReward())
