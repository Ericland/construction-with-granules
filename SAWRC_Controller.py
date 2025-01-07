# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:32:39 2023

@author: ericl
"""

# In[]
import pymunk
from pymunk import Vec2d

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from time import perf_counter 
# from collections import deque
from collections import namedtuple
# import pickle
import os

from World import Wall, Grains 
from SAW_Robot import Robot, Action_Info
from SAW_Modeling_Toolbox import Visualization_by_Pygame
import Utility


# In[]
class Leveling_Controller: 
    def __init__(
            self, 
            simulator, 
            ): 
        self.simulator = simulator 
        self.mode = 'navigation'
        self.counter_leveling = 0 
        
        
    def compute_leveling_pwmRL(self, k): 
        pwmR = np.piecewise(k, [k<-0.2, ((k>=-0.2)&(k<=0)), k>0], [lambda k: -255, lambda k: 1275*k, lambda k: 255]) 
        pwmL = np.piecewise(k, [k<0, ((k>=0)&(k<=0.2)), k>0.2], [lambda k: -255, lambda k: 1275*k, lambda k: 255])
        
        return pwmR, pwmL 
    
    
    def compute_navigation_pwmRL(self, x_destination): 
        xr, yr = self.simulator.robot_info.position 
        xrL, yrL = self.simulator.robot.link_left_list[7].body.position 
        xrR, yrR = self.simulator.robot.link_right_list[7].body.position 
        dR = abs(xrR - xr) 
        dL = abs(xrL - xr) 
        if xr < x_destination-dR:  
            pwmR, pwmL = (255, 255) 
        elif xr > x_destination+dL:  
            pwmR, pwmL = (-255, -255) 
        elif xr >= x_destination-dR and xr <= x_destination+dL: 
            pwmR, pwmL = (0, 0) 
            
        return pwmR, pwmL
    
    
    def plot_system_response(self): 
        k = np.linspace(-0.4, 0.4, num=100) 
        pwmR, pwmL = self.compute_leveling_pwmRL(k)
        fig, ax = plt.subplots()
        fig.dpi = 200 
        ax.plot(k, pwmR, label='pwmR')
        ax.plot(k, pwmL, '--', label='pwmL')
        ax.legend()
        ax.set_xlabel('k')
        ax.set_title('leveling mode')
        
        return fig, ax 
    
    
    def predict(self): 
        if self.mode == 'navigation': 
            wci = self.simulator.robot_collides_with_wall() 
            if wci.right or wci.left: 
                pwmR, pwmL = (0, 0)
                self.mode = 'leveling' 
            else: 
                xo, yo = self.simulator.structure_info.outline # get current terrain 	
                xt, yt = self.simulator.equalize_outline_integral((xo, yo), self.simulator.target_terrain_outline) # get equalized targer terrain
                x_destination = xt[np.argmax(yo-yt)] 
                pwmR, pwmL = self.compute_navigation_pwmRL(x_destination) 
                if pwmR == 0 and pwmL == 0: 
                    self.mode = 'leveling' 
        if self.mode == 'leveling': 
            pwmR_arr, pwmL_arr = self.compute_leveling_pwmRL(np.tan(self.simulator.robot_info.orientation))
            pwmR = int(pwmR_arr)
            pwmL = int(pwmL_arr)
            self.counter_leveling += 1 
            if self.counter_leveling == 50: 
                self.counter_leveling = 0 
                self.mode = 'navigation' 
        
        return pwmR, pwmL 
    
    
# In[]
if __name__ == "__main__": 
    con = Leveling_Controller(None)
    con.plot_system_response()
        








