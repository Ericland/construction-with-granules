# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:38:01 2023

@author: ericl
"""


# In[]
# import pygame
# import pymunk
# import pymunk.autogeometry
# import pymunk.pygame_util
from pymunk import Vec2d

# import random
# random.seed(5)  # try keep difference the random factor the same each run.
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
from collections import deque, namedtuple

from SAW_Kinematics import Robot_Kinematics
from World import PivotJoint, Motor, Chassis, EndEffector
from World import Link, Linkf
from World import get_limit_velocity_func
import Utility


# In[]
Action_Info = namedtuple('Action_Info', ['pwmR', 'pwmL'])
    
    
# In[]
class Robot: 
    def __init__(
            self, 
            name='robot1', 
            link_num=12, 
            link_length=14, 
            link1_length=28, 
            linkf_length=30, 
            helix_L=120, 
            helix_A=22.5, 
            r=10, 
            thr=4, 
            thl=7, 
            r0=13, 
            thr0=4, 
            thl0=7, 
            chassis_h=20, 
            chassis_w=32, 
            ): 
        # robot parameters
        self.type = "Robot"
        self.name = name
        self.link_num = link_num
        self.link_length = link_length
        self.link1_length = link1_length
        self.linkf_length = linkf_length
        self.helix_L = helix_L
        self.helix_A = helix_A
        self.r = r
        self.thr = thr
        self.thl = thl
        self.r0 = r0
        self.thr0 = thr0
        self.thl0 = thl0
        self.chassis_h = chassis_h
        self.chassis_w = chassis_w
        self.width = (self.helix_L+self.linkf_length)*2 + self.chassis_w
        self.height = self.helix_A*2
        
        # default kinematics parameters
        self.freq_max = 1.0 # max frequency allowed 
        self.freqL = 1.0 # frequency of left motor 
        self.freqR = 1.0 # frequency of right motor
        self.dt = 0.1 # time resolution for simulation 
        self.phiR = 0 # phase of right tail (time-varying)
        self.phiL = 0 # phase of left tail (time-varying)
        
        # state parameters
        self.robot_added = False
        self.robot_is_on = False
        self.correct_sim_errors = False
        
        # action parameters
        self.action_num = 0
        self.action_history = deque(maxlen=50)
        self.action_dict = { # all robot actions 
            -1: "stay", 
            0: "move_right", 
            1: "move_left", 
            2: "dig", 
            3: "pile_right", 
            4: "pile_left", 
            }
        
        # error correction parameters
        self.diff_thetaR_history = deque(maxlen=50)
        self.diff_thetaL_history = deque(maxlen=50)
        
        
    def get_kinematics(self, side='right'): 
        """
        Import kinematics solvers

        Returns
        -------
        None.

        """
        if side == 'right': 
            # kinematics of right tail
            self.sawkR = Robot_Kinematics(
                self.link_num, 
                self.link_length, 
                self.link1_length, 
                self.helix_L, 
                self.helix_A, 
                self.freqR, 
                self.dt, 
                self.phiR, 
                right_tail=True, 
                advancing_wave=True, 
                )
        elif side == 'left': 
            # kinematics of left tail
            self.sawkL = Robot_Kinematics(
                self.link_num, 
                self.link_length, 
                self.link1_length, 
                self.helix_L, 
                self.helix_A, 
                self.freqL, 
                self.dt, 
                self.phiL, 
                right_tail=False, 
                advancing_wave=True, 
                )
        
        
    def get_pv_for_links(self, ksolR, ksolL, offset_right, offset_left): 
        """
        Compute initial configuration of links
        pv_list = [(pi, vi)]: 
            pi is the start point of the link
            vi is the vector representing the link starting from origin
        offset is the offset for the starting point of the right tail 

        Returns
        -------
        pv_right_list : TYPE
            right tail 
        pv_left_list : TYPE
            left tail 

        """
        # get initial end points of links
        ep = list(ksolR['xy'].iloc[:, 0]) # initial end points 
        size = len(ep)
        epx = np.array([ee[0] for ee in ep]) + offset_right[0]
        epy = np.array([ee[1] for ee in ep]) + offset_right[1]
        vx = epx[1:] - epx[0:size-1]
        vy = epy[1:] - epy[0:size-1]
        
        # compute pv for right tail
        pv_right_list = []
        for ii in range(size-2): 
            pv_right_list.append((Vec2d(epx[ii], epy[ii]), Vec2d(vx[ii], vy[ii])))
        sf = self.linkf_length / self.link_length
        pv_right_list.append((Vec2d(epx[ii+1], epy[ii+1]), Vec2d(vx[ii+1]*sf, vy[ii+1]*sf)))
            
        # compute pv for left tail
        ep = list(ksolL['xy'].iloc[:, 0]) # initial end points
        size = len(ep)
        epx = np.array([ee[0] for ee in ep]) + offset_left[0]
        epy = np.array([ee[1] for ee in ep]) + offset_left[1]
        vx = epx[1:] - epx[0:size-1]
        vy = epy[1:] - epy[0:size-1]
        pv_left_list = []
        for ii in range(size-2): 
            pv_left_list.append((Vec2d(epx[ii], epy[ii]), Vec2d(vx[ii], vy[ii])))
        pv_left_list.append((Vec2d(epx[ii+1], epy[ii+1]), Vec2d(vx[ii+1]*sf, vy[ii+1]*sf)))
            
        return pv_right_list, pv_left_list
        
        
    def add_robot(self, space, pos): 
        """
        Add robot to pymunk space

        Parameters
        ----------
        space : TYPE
            DESCRIPTION.
        pos : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # compute initial kinematics
        self.get_kinematics(side='right')
        self.get_kinematics(side='left')
        ksolR = self.sawkR.compute_kinematics_over_dt()
        ksolL = self.sawkL.compute_kinematics_over_dt()
        for kk in ['Theta', 'theta', 'xy']: 
            ksolR[kk].iloc[:, 1] = ksolR[kk].iloc[:, 0]
            ksolL[kk].iloc[:, 1] = ksolL[kk].iloc[:, 0]
        ksolR['dtheta'].iloc[:, 0] = 0
        ksolL['dtheta'].iloc[:, 0] = 0
        
        # store all pymunk body objects added to the robot
        self.body_list = []
        
        # add chassis
        self.chassis = Chassis(space, pos, self.chassis_h, self.chassis_w)
        self.body_list.append(self.chassis.body)
        
        # get pv for tails
        offset_right = pos + (self.chassis_w/2, 0)
        offset_left = pos - (self.chassis_w/2, 0)
        pv_right_list, pv_left_list = self.get_pv_for_links(ksolR, ksolL, offset_right, offset_left)
        self.link_right_list = []
        self.motor_right_list = []
        self.link_left_list = []
        self.motor_left_list = []
        
        # add 1st link of right tail
        p0, v0 = pv_right_list[0]
        link0 = Link(space, p0, v0, self.r0, self.thr0, self.thl0)
        self.link_right_list.append(link0)
        PivotJoint(space, self.chassis.body, link0.body, (self.chassis_w/2, 0), (0, 0))
        self.motor_right_list.append(Motor(space, self.chassis.body, link0.body, tag=0))
        # add rest links of right tail
        pv_right_list.pop(0)
        for ii in range(len(pv_right_list)): 
            pi, vi = pv_right_list[ii]
            linklast = self.link_right_list[-1]
            if ii+1 == len(pv_right_list): 
                linki = Linkf(space, pi, vi, self.r)
            else: 
                linki = Link(space, pi, vi, self.r, self.thr, self.thl)
            self.link_right_list.append(linki)
            PivotJoint(space, linklast.body, linki.body, linklast.v, (0, 0))
            self.motor_right_list.append(Motor(space, linklast.body, linki.body, tag=ii+1))
        for ll in self.link_right_list: 
            self.body_list.append(ll.body)
            
        # add end effector of right tail
        # end effector must be placed at the position where the vector of last link points to
        # otherwise simulation error correction will go wrong 
        linklast = self.link_right_list[-1]
        self.eeR = EndEffector(space, linklast.pos_init+linklast.v, True)
        PivotJoint(space, linklast.body, self.eeR.body, linklast.v, (0, 0))
        PivotJoint(space, linklast.body, self.eeR.body, (0, 0), (0, 0))
        self.body_list.append(self.eeR.body)
            
        # add 1st link of left tail
        p0, v0 = pv_left_list[0]
        link0 = Link(space, p0, v0, self.r0, self.thr0, self.thl0)
        self.link_left_list.append(link0)
        PivotJoint(space, self.chassis.body, link0.body, (-self.chassis_w/2, 0), (0, 0))
        self.motor_left_list.append(Motor(space, self.chassis.body, link0.body, tag=0))
        # add rest links of left tail
        pv_left_list.pop(0)
        for ii in range(len(pv_left_list)):
            pi, vi = pv_left_list[ii]
            linklast = self.link_left_list[-1]
            if ii+1 == len(pv_left_list): 
                linki = Linkf(space, pi, vi, self.r)
            else:
                linki = Link(space, pi, vi, self.r, self.thr, self.thl)
            self.link_left_list.append(linki)
            PivotJoint(space, linklast.body, linki.body, linklast.v, (0, 0))
            self.motor_left_list.append(Motor(space, linklast.body, linki.body, tag=ii+1))
        for ll in self.link_left_list: 
            self.body_list.append(ll.body)
            
        # add end effector of left tail
        # end effector must be placed at the position where the vector of last link points to
        # otherwise simulation error correction will go wrong 
        linklast = self.link_left_list[-1]
        self.eeL = EndEffector(space, linklast.pos_init+linklast.v, True)
        PivotJoint(space, linklast.body, self.eeL.body, linklast.v, (0, 0))
        PivotJoint(space, linklast.body, self.eeL.body, (0, 0), (0, 0))
        self.body_list.append(self.eeL.body)
            
        self.robot_added = True
        
        return ksolR, ksolL
    
    
    def limit_body_velocity(self, max_body_velocity): 
        """
        Limit the velocity of every body to avoid simulation errors
        CAUTION: using this function may deteriorate simulation efficiency!!!

        Parameters
        ----------
        max_body_velocity : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.robot_added: 
            limit_velocity_func = get_limit_velocity_func(max_body_velocity)
            for body in self.body_list: 
                body.velocity_func = limit_velocity_func
        
        
    def compute_motor_rate(self, ksol): 
        """
        Compute motor rate based on change of dtheta

        Parameters
        ----------
        ksol : output from kinematics solver
            DESCRIPTION.

        Returns
        -------
        None.

        """
        motor_rate = ksol['dtheta'].to_numpy() / self.dt * -1
        
        return(motor_rate)
    
    
    def get_position(self): 
        if self.robot_added: 
            pos = self.chassis.body.position
        else:
            pos = Vec2d(0, 0)
        
        return pos
    
    
    def get_end_effector_position(self, side='right'): 
        if self.robot_added: 
            if side == 'right': 
                pos = self.eeR.body.position
            elif side == 'left': 
                pos = self.eeL.body.position
        else:
            pos = Vec2d(0, 0)
        
        return pos
    
    
    def get_orientation(self): 
        if self.robot_added: 
            vec = self.link_right_list[0].body.position - self.link_left_list[0].body.position
            orientation = vec.angle
        else:
            orientation = 0.0
        
        return orientation
    
    
    def get_orientation_vector(self): 
        if self.robot_added: 
            vec = self.link_right_list[0].body.position - self.link_left_list[0].body.position
            vecn = vec.normalized()
            
        return vecn
    
    
    def get_displacement(self): 
        """
        Compute displacement

        Returns
        -------
        None.

        """
        if self.robot_added and self.robot_is_on: 
            dism = self.get_position() - self.pos_init
        else:
            dism = Vec2d(0, 0)
            
        return dism
    
    
    def get_moving_velocity(self): 
        if self.robot_added: 
            vel = self.chassis.body.velocity.length
        else:
            vel = 0
        
        return vel
    
    
    def get_max_body_velocity(self): 
        """
        Get the max velocity for every bodies attached to the robot

        Returns
        -------
        None.

        """
        if self.robot_added: 
            velocity_list = []
            for body in self.body_list: 
                velocity_list.append(body.velocity.length)
            max_body_velocity = np.amax(velocity_list)
        else:
            max_body_velocity = 0
            
        return max_body_velocity
    
    
    def get_link_xy(self): 
        """
        Get end points of links

        Returns
        -------
        xyR : list
            [Vec2d(xi, yi)]
        xyL : list
            [Vec2d(xi, yi)]

        """
        xyR = [self.chassis.body.position]
        # An improvised way to find first point of xyL
        # It might be worth investigating why such way can fix the problem 
        # It seems that dtheta for first link of left tail from the solution needs to be adjusted
        xyL = [self.link_left_list[0].body.position*2 - self.chassis.body.position] 
        for ii in range(self.link_num + 1): 
            xyR.append(self.link_right_list[ii].body.position)
            xyL.append(self.link_left_list[ii].body.position)
        xyR.append(self.eeR.body.position)
        xyL.append(self.eeL.body.position)
        
        return xyR, xyL
    
    
    def compute_link_config(self, xy):
        """
        Compute link configurations based on link end points. 
        This function computes: 
            abs link angles
            relative link angles

        Parameters
        ----------
        xy : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        rk = Robot_Kinematics()
        vecs = rk.link2vec(xy)
        orientation_angle = self.get_orientation()
        
        # get absolute angle 
        Theta_list = rk.compute_abs_link_angles(vecs)
        Theta_list.pop(0)
        Theta = Utility.rad2pipi(np.array(Theta_list) - orientation_angle)
        
        # get relative angle 
        theta_list = rk.compute_relative_link_angles(vecs)
        theta_list.pop(0)
        theta = np.array(theta_list)
            
        return Theta, theta
    
    
    def check_link_angles(self, ksolR, ksolL):
        """
        Compare current abs/relative link angles to the desired ones

        Parameters
        ----------
        ksolR : TYPE
            DESCRIPTION.
        ksolL : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert self.robot_added, "robot needs to be added first"
        
        # get current link configuration
        xyR, xyL = self.get_link_xy()
        ThetaR, thetaR = self.compute_link_config(xyR)
        ThetaL, thetaL = self.compute_link_config(xyL)
        
        # get desired abs and relative link angles
        ThetaR_goal = ksolR['Theta'].iloc[:, 1].to_numpy()
        ThetaL_goal = ksolL['Theta'].iloc[:, 1].to_numpy()
        thetaR_goal = ksolR['theta'].iloc[:, 1].to_numpy()
        thetaL_goal = ksolL['theta'].iloc[:, 1].to_numpy()
        
        # compute the difference in abs link angles
        self.diff_ThetaR = Utility.rad2pipi(ThetaR_goal - ThetaR)
        self.diff_ThetaL = Utility.rad2pipi(ThetaL_goal - ThetaL)
        diff_ThetaRL = np.concatenate((self.diff_ThetaR, self.diff_ThetaL))
        
        # compute the difference in relative link angles
        self.diff_thetaR = Utility.rad2pipi(thetaR_goal - thetaR)
        self.diff_thetaL = Utility.rad2pipi(thetaL_goal - thetaL)
        diff_thetaRL = np.concatenate((self.diff_thetaR, self.diff_thetaL))
        self.diff_thetaR_history.append(self.diff_thetaR)
        self.diff_thetaL_history.append(self.diff_thetaL)
        
        return diff_ThetaRL, diff_thetaRL
    
    
    def _compute_cmr(self): 
        """
        Compute the correctional motor rate for compensating errors

        Parameters
        ----------
        last_action : str
            DESCRIPTION.
        method : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        Kp = 1
        cmrL = Utility.rad2pipi(self.diff_thetaL*Kp) / self.dt * -1
        cmrR = Utility.rad2pipi(self.diff_thetaR*Kp) / self.dt * -1
                
        return cmrR, cmrL
    
    
    def correct_link_angles(self): 
        """
        Correct the abs link angles by modifying the motor rate

        Parameters
        ----------
        diff_thetaR : numpy array
            desired relative link angles of right tail - current ones
        diff_thetaL : TYPE
            desired relative link angles of left tail - current ones

        Returns
        -------
        None.

        """
        assert self.robot_is_on, "robot needs to be activated first"
        
        if self.correct_sim_errors: 
            # compute how much the motor should rotate more to compensate the error
            cmrR, cmrL = self._compute_cmr() 
            for ii in range(self.link_num+1): 
                self.motor_left_list[ii].modify_rate(cmrL[ii])
                self.motor_right_list[ii].modify_rate(cmrR[ii])
                
            # turn off correction flag when correction is complete
            self.correct_sim_errors = False
            
        else:
            cmrR = np.zeros(self.link_num+1)
            cmrL = np.zeros(self.link_num+1)
            
        return np.concatenate((cmrR, cmrL))
    
    
    def report_motor_rate(self): 
        mrR_list = []
        mrL_list = []
        for ii in range(self.link_num+1): 
            rR = self.motor_right_list[ii].report_rate()
            rL = self.motor_left_list[ii].report_rate()
            mrR_list.append(rR)
            mrL_list.append(rL)
        mrR = np.array(mrR_list)
        mrL = np.array(mrL_list)
        
        return np.concatenate((mrR, mrL))
    
    
    def _update_motor_rate(self, mrR, mrL): 
        """
        Update motor rate without changing the phase
        Use cautiously!
        """
        for ii in range(self.link_num+1): 
            self.motor_right_list[ii].update_rate(mrR[ii])
            self.motor_left_list[ii].update_rate(mrL[ii])
            
            
    def report_motor_force(self, dt): 
        """ 
        dt: input passed to space.step()
        """
        mfR_list = [] 
        mfL_list = []
        for ii in range(self.link_num+1): 
            mfR_list.append(self.motor_right_list[ii].motor.impulse/dt)
            mfL_list.append(self.motor_left_list[ii].motor.impulse/dt)
        mfRL_list = mfR_list + mfL_list 
        
        return np.array(mfRL_list)
    
    
    def activate(self): 
        """
        Activate robot 

        Returns
        -------
        None.

        """
        assert self.robot_added, "robot needs to be added first"
        
        # record initial position
        self.pos_init = self.get_position()
        self.orientation_init = self.get_orientation()
        self.ovec_init = self.get_orientation_vector()
        
        self.robot_is_on = True
        
        
    def is_static(self, relaxed_condition=True): 
        """
        Check if the robot is static

        Returns
        -------
        None.

        """
        if self.robot_added: 
            robot_is_static = True
            for body in self.body_list: 
                pass_test = True
                if relaxed_condition: 
                    if body.velocity.length > 1: 
                        pass_test = False
                else:
                    if not body.is_sleeping: 
                        pass_test = False
                if not pass_test:
                    robot_is_static = False
                    break
        else:
            robot_is_static = False
            
        return robot_is_static
    
    
    def take_action(self, pwmR, pwmL): 
        """
        Parameters
        ----------
        pwmL : int
            -255 ~ 255
        pwmR : int
            -255 ~ 255

        Returns
        -------
        None.

        """
        assert self.robot_is_on, "robot needs to be activated first"

        # right motors 
        if pwmR != 0: 
            # solve kinematics 
            self.freqR = abs(pwmR/255) * self.freq_max
            self.get_kinematics(side='right')
            if pwmR >= 0: 
                self.sawkR.advancing_wave = True
            else:
                self.sawkR.advancing_wave = False
            ksolR = self.sawkR.compute_kinematics_over_dt()
            # actuate motors
            mrR = self.compute_motor_rate(ksolR)
            for ii in range(self.link_num + 1): 
                self.motor_right_list[ii].update_rate(mrR.flat[ii])
            # update phases
            self.phiR -= self.sawkR.w * self.dt
        else:
            # solve kinematics 
            self.freqR = 1
            self.get_kinematics(side='right')
            ksolR = self.sawkR.compute_kinematics_over_dt()
            for kk in ['Theta', 'theta', 'xy']: 
                ksolR[kk].iloc[:, 1] = ksolR[kk].iloc[:, 0]
            ksolR['dtheta'].iloc[:, 0] = 0
            # actuate motors 
            for ii in range(self.link_num + 1): 
                self.motor_right_list[ii].deactivate()
                
        # left motors 
        if pwmL != 0: 
            # solve kinematics 
            self.freqL = abs(pwmL/255) * self.freq_max
            self.get_kinematics(side='left')
            if pwmL >= 0: 
                self.sawkL.advancing_wave = True
            else:
                self.sawkL.advancing_wave = False
            ksolL = self.sawkL.compute_kinematics_over_dt()
            # actuate motors
            mrL = self.compute_motor_rate(ksolL)
            for ii in range(self.link_num + 1): 
                self.motor_left_list[ii].update_rate(mrL.flat[ii])
            # update phases
            self.phiL -= self.sawkL.w * self.dt
        else:
            # solve kinematics 
            self.freqL = 1
            self.get_kinematics(side='left')
            ksolL = self.sawkL.compute_kinematics_over_dt()
            for kk in ['Theta', 'theta', 'xy']: 
                ksolL[kk].iloc[:, 1] = ksolL[kk].iloc[:, 0]
            ksolL['dtheta'].iloc[:, 0] = 0
            # actuate motors 
            for ii in range(self.link_num + 1): 
                self.motor_left_list[ii].deactivate()
        
        # update action history
        self.action_num += 1
        af = Action_Info(pwmR, pwmL)
        self.action_history.append(af)
        
        return ksolR, ksolL
        
        

    
