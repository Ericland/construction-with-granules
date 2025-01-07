# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:31:08 2023

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


# In[data classes]
Wall_Collision_Info = namedtuple('Wall_Collision_Info', ['left', 'right'])
Structure_Info = namedtuple('Structure_Info', ['outline'])
Robot_Info = namedtuple('Robot_Info', ['position', 'orientation', 'pwmRL', 'displacement', 'velocity_average'])
Simulation_Info = namedtuple('Simulation_Info', ['simTime', 'runTime', 'simulation_efficiency'])
Construction_Data = namedtuple('Construction_Data', ['structure_info_list', 'robot_info_list', 'simulation_info_list'])


# In[basic functions]
def detect_collision(shape_a, shape_b): 
    cps = shape_a.shapes_collide(shape_b)
    if len(cps.points) > 0: 
        collision = True 
    else:
        collision = False
    
    return collision 


# In[]
class Simulator: 
    def __init__(
            self, 
            initial_terrain_info, 
            target_terrain_info, 
            grain_density=1, 
            grain_friction=100, 
            grain_elasticity=0, 
            random_seed=0, 
            step_num=100, # this number should always be fixed 
            terrain_shape_sampling_resolution=10, # unit=mm
            steepness_max=np.tan(np.radians(54)), # used for outline outlier detection 
            pygame_visualization=True, # simulation visualization 
            save_pygame_screenshot=False, # save visualization image 
            simulation_acceleration=True, 
            simulation_error_analysis=False, # analyze simulation errors
            print_simulation_msg=True, # print tips for simulation progress
            data_autosave=False, # automatically save construction data 
            ): 
        # simulation option parameters 
        self.initial_terrain_info = initial_terrain_info
        self.target_terrain_info = target_terrain_info
        assert initial_terrain_info.space_size[1] == target_terrain_info.space_size[1], "terrain width mismatch!"
        self.space_h, self.space_w = initial_terrain_info.space_size
        self.grain_density = grain_density
        self.grain_friction = grain_friction 
        self.grain_elasticity = grain_elasticity 
        self.random_seed = random_seed
        self.step_num = step_num
        self.steepness_max = steepness_max
        self.terrain_shape_sampling_resolution = terrain_shape_sampling_resolution 
        self.terrain_shape_sampling_points = np.arange(0, self.space_w+terrain_shape_sampling_resolution, terrain_shape_sampling_resolution)
        self.target_terrain_outline = self.sample_outline(self.target_terrain_info.goal_terrain)
        self.target_terrain_goal_terrain = self.equalize_outline_integral(self.initial_terrain_info.goal_terrain, self.target_terrain_info.goal_terrain)
        self.pygame_visualization = pygame_visualization 
        self.save_pygame_screenshot = save_pygame_screenshot
        self.simulation_acceleration = simulation_acceleration
        self.simulation_error_analysis = simulation_error_analysis
        self.print_simulation_msg = print_simulation_msg
        self.data_autosave = data_autosave
        
        # set up pymunk space
        self.space = pymunk.Space()
        self.space.gravity = 0, -9807
        
        # Time a group of bodies must remain idle in order to fall asleep.
        # The sleeping algorithm is disabled when this parameter is inf (the default value)
        # It was found that changing this value will freeze the simulation
        # Thus this parameter should always remain the default value
        # self.space.sleep_time_threshold = 0.01 
        
        # max overlap between shape that is allowed (default: 0.1), higher value improves stability
        self.space.collision_slop = 1
        
        # initialize wall
        self.wall = Wall(self.space, self.space_h, self.space_w)
        
        # initialize grains 
        self.grains = Grains(
            container_height=self.space_h,  
            container_width=self.space_w, 
            grain_density=self.grain_density, 
            grain_friction=self.grain_friction, 
            grain_elasticity=self.grain_elasticity, 
            random_seed=self.random_seed, 
            )
        
        # initialize robot
        self.robot = Robot()
        
        # initialize visualizer 
        if self.pygame_visualization: 
            self.initialize_visualizer()
            
        # simulation state parameter
        self.site_is_prepared = False
        self.simulator_is_on = True 
            
        # simulation data logging parameter
        self.robot_info = None
        self.update_structure_info = True 
        self.structure_info = None
        self.simulation_info = None 
        self.initial_robot_info = None
        self.initial_structure_info = None
        self.initial_simulation_info = None 
        self.initial_terrain_compactness = None 
        # for simulation error analysis 
        self.diff_ThetaRL_list = [] # errors in absolute link angles
        self.diff_thetaRL_list = [] # errors in relative link angles
        self.cmrRL_list = [] # motor rate for compensating the errors 
        self.mrRL_list = [] # motor rate after error correction 
        # for data auto saving 
        self.data_autosave_size = 100 # size of each batch
        self.robot_info_list = []
        self.structure_info_list = []
        self.simulation_info_list = []
        self.data_file_dir_list = [] # directory of data files 
        self.saved_construction_data_dir = None 
        # fata simulation errors 
        self.fatal_error_msg_list = [] 
            
        # debugging functions
        # input of debugging functions: (space, wall, grains, robot)
        self.debugging_mode_is_on = False
        self.pre_step_func = None # this function runs before space.step() runs
        self.in_step_func = None # this function runs as space.step() runs
        self.post_step_func = None # this function runs after space.step() runs
        self.debugging_data_list = [] # this list gathers returned data from debugging function 
            
        # initialize time parameter
        self.reset_timer()
        
        
    def initialize_visualizer(self): 
        self.visualizer = Visualization_by_Pygame((self.space_w, self.space_h), 100) 
        self.pygame_visualization = True 
        
        
    def reset_timer(self): 
        self.stepCounter = 0
        self.simTime = 0
        self.runTime = 1e-6
        self.runTimeStart = perf_counter()
        
        
    def update_timer(self): 
        self.stepCounter += 1
        self.simTime = self.stepCounter * self.robot.dt
        self.runTime = perf_counter() - self.runTimeStart
        
        
    def sample_outline(self, outline): 
        xo, yo = outline
        xos = np.copy(self.terrain_shape_sampling_points)
        yos = np.interp(xos, xo, yo)
        
        return xos, yos
        
        
    def get_structure_info(self): 
        outline = self.grains.get_outline(self.space, plot=0) 
        outline_sampled = self.sample_outline(outline)
        structure_info = Structure_Info(outline_sampled) 
        
        return structure_info
    
    
    def get_robot_info(self): 
        position = self.robot.get_position()
        orientation = self.robot.get_orientation()
        if len(self.robot.action_history) > 0: 
            pwmRL = self.robot.action_history[-1]
        else:
            pwmRL = Action_Info(0, 0)
        dism = self.robot.get_displacement()
        vel_avg = dism / (self.simTime + 0.001)
        robot_info = Robot_Info(position, orientation, pwmRL, dism, vel_avg)
        
        return robot_info 
    
    
    def get_simulation_info(self): 
        simulation_info = Simulation_Info(self.simTime, self.runTime, self.simTime/self.runTime)
        
        return simulation_info
            
            
    def get_robot_drop_location(self, xdrop_desired=0): 
        # get xdrop
        if xdrop_desired - self.robot.width/2 - 10 < 0: 
            xdrop = self.robot.width/2 + self.robot.chassis_w
        elif xdrop_desired + self.robot.width/2 + 10 > self.space_w:
            xdrop = self.space_w - self.robot.width/2 - self.robot.chassis_w
        else:
            xdrop = xdrop_desired
        
        # get ydrop
        xo, yo = self.grains.get_outline(self.space, plot=0)
        xr_list = np.linspace(xdrop-self.robot.width/2, xdrop+self.robot.width/2)
        yor_list = np.interp(xr_list, xo, yo)
        ydrop = np.amax(yor_list) + self.robot.height
    
        return xdrop, ydrop
    
    
    def prepare_site(self, x_robot_drop=0, initial_robot_phase=0): 
        if self.simulator_is_on: 
            # form the given terrain 
            self.grains.form_terrain(self.space, self.initial_terrain_info)
            
            # simulation parameter 
            grains_settled = False
            robot_settled = False
            
            if self.print_simulation_msg: print('Begin site preparation')
            
            while True: 
                # check pygame screen input 
                if self.pygame_visualization: 
                    event_msg = self.visualizer.check_event() # check user input
                    if event_msg == "stop": 
                        raise Exception('Site preparation fails!')
                
                # Check if grains are settled
                if not grains_settled: 
                    if self.grains.is_static(True): 
                        grains_settled = True
                        xt, yt = self.grains.get_outline(self.space, plot=0)
                        self.initial_terrain_compactness = self.grains.get_compactness(self.space)
                        
                # If robot has not been added, add robot if grains are settled
                if grains_settled and not self.robot.robot_added: 
                    xo, yo = self.grains.get_outline(self.space, plot=0)
                    xdrop, ydrop = self.get_robot_drop_location(x_robot_drop)
                    p_add_robot = Vec2d(xdrop, ydrop)
                    self.robot.phiL = initial_robot_phase
                    ksolR, ksolL = self.robot.add_robot(self.space, p_add_robot)
                
                # If robot is added, check if robot is settled
                if self.robot.robot_added: 
                    if self.robot.is_static(True): 
                        robot_settled = True
                        
                # if robot is settled and robot has not been activated, activate it and finish site preparation
                if robot_settled and not self.robot.robot_is_on: 
                    self.wall.add_lid(self.space) # add lid to the container
                    self.robot.activate()
                    if self.print_simulation_msg: print('Site preparation is done')
                    self.site_is_prepared = True 
                    # get initial state info 
                    self.initial_structure_info = self.get_structure_info()
                    self.initial_robot_info = self.get_robot_info()
                    self.initial_simulation_info = self.get_simulation_info()
                    break 
                        
                # =========================================================================
                # Update pymunk simulation 
                # All actions become effective after this! 
                # =========================================================================
                for ii in range(self.step_num): 
                    self.space.step(self.robot.dt / self.step_num)
                self.update_timer()
                
                # Get state info
                simulation_info = self.get_simulation_info()
                display_text = dict(simulation_info._asdict())
                
                # update the visualizer
                if self.pygame_visualization and self.simulator_is_on: 
                    self.visualizer.update(self.space, display_text)
                
                # =========================================================================
                # when following conditions are met, stop simulation 
                # =========================================================================
                # stop simulation after 100s
                if self.simTime > 100: 
                    raise Exception('Site preparation fails!')
            
            # reset time
            self.reset_timer() 
            
            
    def detect_simulation_explosion(self): 
        '''
        Currently not being used 
        CAUTION: using this function may deteriorate simulation efficiency!!!
        '''
        # if max grain velocity > threshold, report fatal error 
        max_grain_velocity = np.amax(self.report_grain_velocity())
        detection_threshold = 1e+8 
        if max_grain_velocity >= detection_threshold:  
            error_msg = 'Fatal Error: simulation explosion!' 
            self.fatal_error_msg_list.append(error_msg) 
            
            
    def construct(self, pwmR, pwmL, **kwargs): 
        if self.simulator_is_on and self.site_is_prepared: 
            # debugging
            if self.debugging_mode_is_on: 
                self.debugging_data_list = [] # initialize debugging data list 
            
            # check pygame screen input 
            if self.pygame_visualization: 
                event_msg = self.visualizer.check_event() # check user input
                if event_msg == "stop": 
                    self.stop_simulation()
                    
            # set motor rates 
            ksolR, ksolL = self.robot.take_action(pwmR, pwmL)
            
            # correct errors 
            cmrRL = self.robot.correct_link_angles() # Correct simulation errors 
            if self.simulation_error_analysis: 
                self.cmrRL_list.append(cmrRL)
                self.mrRL_list.append(self.robot.report_motor_rate()) # log motor rates
            
            # accelerate simulation 
            if self.simulation_acceleration: 
                self.grains.sleep_selectively(self.space, self.robot, scale=1, print_msg=False)
                
            # debugging
            if self.debugging_mode_is_on: 
                if self.pre_step_func != None: 
                    debugging_data = self.pre_step_func(self.space, self.wall, self.grains, self.robot)
                    if debugging_data != None: 
                        self.debugging_data_list.append(debugging_data)
                    
            # =========================================================================
            # Update pymunk simulation 
            # All actions become effective after this! 
            # =========================================================================
            for ii in range(self.step_num): 
                self.space.step(self.robot.dt / self.step_num)
                # debugging
                if self.debugging_mode_is_on: 
                    if self.in_step_func != None: 
                        debugging_data = self.in_step_func(self.space, self.wall, self.grains, self.robot)
                        if debugging_data != None: 
                            self.debugging_data_list.append(debugging_data)
                # detect simulation explosion 
                # this function is disabled due to the deterioration of simulation efficiency 
                # self.detect_simulation_explosion()
            self.update_timer()
            
            # Check outcome of robot actions and correct errors if necessary
            if self.robot.robot_is_on: 
                diff_ThetaRL, diff_thetaRL = self.robot.check_link_angles(ksolR, ksolL)
                self.robot.correct_sim_errors = True # raise flag so that sim errors can be corrected in the next loop
                if self.simulation_error_analysis: 
                    self.diff_ThetaRL_list.append(diff_ThetaRL) 
                    self.diff_thetaRL_list.append(diff_thetaRL)
                    
            # Get state info 
            if self.update_structure_info: 
                self.structure_info = self.get_structure_info() # check errors as well 
            self.robot_info = self.get_robot_info()
            self.simulation_info = self.get_simulation_info()
            
            # update the visualizer
            if self.pygame_visualization and self.simulator_is_on: 
                display_text = dict(self.simulation_info._asdict())
                display_text.update(dict(self.robot_info._asdict()))
                other_info = Utility.get_kwargs(kwargs, 'other_info', None) 
                if other_info != None: 
                    display_text.update(other_info) 
                self.visualizer.update(self.space, display_text)
                if self.save_pygame_screenshot: 
                    ps_save_loc = 'img/simulation_screenshot/'
                    ps_file_name = 'img_' + str(self.stepCounter) + '.png' 
                    self.visualizer.save_screenshot(save_loc=ps_save_loc, file_name=ps_file_name)
            
            # debugging
            if self.debugging_mode_is_on: 
                if self.post_step_func != None: 
                    debugging_data = self.post_step_func(self.space, self.wall, self.grains, self.robot)
                    if debugging_data != None: 
                        self.debugging_data_list.append(debugging_data)
            
            # save data in batch automatically 
            if self.data_autosave: 
                self.structure_info_list.append(self.structure_info)
                self.robot_info_list.append(self.robot_info)
                self.simulation_info_list.append(self.simulation_info)
                if len(self.structure_info_list) == self.data_autosave_size: 
                    # save data 
                    construction_data = Construction_Data(
                        self.structure_info_list, 
                        self.robot_info_list, 
                        self.simulation_info_list, 
                        )
                    file_name = 'cd_' + str(len(self.data_file_dir_list)) + '.pkl'
                    data_file_dir = Utility.save_data(
                        construction_data, 
                        save_loc='data/autosave/', 
                        file_name=file_name, 
                        print_msg=False, 
                        )
                    self.data_file_dir_list.append(data_file_dir)
                    # clear list 
                    self.structure_info_list = []
                    self.robot_info_list = []
                    self.simulation_info_list = [] 
             
            #################### CAUTION!!! ####################
            # stop simulation if fatal simulation error occurs 
            self.fatal_error_msg_list += self.grains.fatal_error_msg_list # add fatal errors reported from grains
            if len(self.fatal_error_msg_list) > 0: 
                self.stop_simulation()
                # plot current state to help debugging 
                for error_msg in self.fatal_error_msg_list: 
                    print(error_msg)
                self.plot_construction(self.structure_info, self.robot_info, self.simulation_info)
                self.plot_grain_particles()
                    
                    
    def collect_construction_data(self, delete_data_batches=True): 
        if self.data_autosave and len(self.data_file_dir_list) > 0: 
            # merge all data 
            all_structure_info_list = [self.initial_structure_info]
            all_robot_info_list = [self.initial_robot_info]
            all_simulation_info_list = [self.initial_simulation_info]
            for data_file_dir in self.data_file_dir_list: 
                save_loc, file_name = data_file_dir
                cd = Utility.load_data(save_loc, file_name)
                all_structure_info_list += cd.structure_info_list
                all_robot_info_list += cd.robot_info_list
                all_simulation_info_list += cd.simulation_info_list
            # save merged data 
            construction_data = Construction_Data(
                all_structure_info_list, 
                all_robot_info_list, 
                all_simulation_info_list, 
                )
            self.saved_construction_data_dir = Utility.save_data(
                construction_data, 
                save_loc='data/autosave/', 
                file_name='default', 
                print_msg=True, 
                )
            # delete original data batches
            if delete_data_batches: 
                for data_file_dir in self.data_file_dir_list: 
                    save_loc, file_name = data_file_dir
                    os.remove(save_loc + file_name)
                self.data_file_dir_list = []
                print('Data batches are deleted!')
        
        
    def stop_simulation(self): 
        if self.simulator_is_on: 
            # collect data 
            if self.data_autosave: 
                self.collect_construction_data()
            
            # stop visualization
            if self.pygame_visualization: 
                self.visualizer.stop()
                
            # print simulation information at end state
            if self.simulation_info != None and self.robot_info != None: 
                display_text = dict(self.simulation_info._asdict())
                display_text.update(dict(self.robot_info._asdict()))
                if self.print_simulation_msg: 
                    for kk, vv in display_text.items(): print(kk, ': ', vv)
                    print('Robot construction simulation stops')
            
            self.simulator_is_on = False
            
            
    def equalize_outline_integral(self, outline_target, outline_input): 
        """
        Output an outline that has equal integral with the outline_target
        """
        xt, yt = outline_target # get input outline
        xo, yo = outline_input # get current terrain 	
        assert np.all(xt==xo), "x-axis must match!"	
        offset = (np.trapz(yt, x=xt) - np.trapz(yo, x=xt)) / (np.amax(xt) - np.amin(xt))	
        yoe = yo + offset # equalize the integral of the terrain shape function
        outline_equalized = (xt, yoe) 

        return outline_equalized         
            
    
    def get_W1_distance(self, structure_info, plot=False):	
        """	
        Get the W1 distance between the target terrain and the terrain outline in structure_info	
        """	
        xo, yo = structure_info.outline # get current terrain 	
        xt, yt = self.equalize_outline_integral((xo, yo), self.target_terrain_outline) # get equalized targer terrain
        po = yo / np.sum(yo) # convert terrain to probability mass function 	
        pt = yt / np.sum(yt) # convert terrain to probability mass function 	
        W1 = Utility.compute_W1_distance(po, pt, xt) # compute W1 distance 	
        if plot: 	
            fig, ax = plt.subplots()	
            fig.dpi = 200	
            ax.plot(xt, yt, '.')	
            ax.plot(xo, yo, '.')	
            print(np.trapz(yo, x=xo), np.trapz(yt, x=xt))	
        	
        return W1
    
    
    def get_W1_distance_all(self, construction_data): 
        W1_list = []
        for structure_info in construction_data.structure_info_list: 
            W1_list.append(self.get_W1_distance(structure_info)) 
        
        return W1_list
    
    
    def plot_W1_distance(self, construction_data, **kwargs): 
        W1_list = self.get_W1_distance_all(construction_data)
        subplots = Utility.get_kwargs(kwargs, 'subplots', None) 
        if subplots == None: 
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots
        fig.dpi = 200
        ax.set_title('W1 distance')
        ax.set_xlabel('time')
        ax.plot(W1_list)
        
        return fig, ax 
    
            
    def plot_construction(self, structure_info, robot_info, simulation_info, **kwargs):
        xlim = Utility.get_kwargs(kwargs, 'xlim', None)
        ylim = Utility.get_kwargs(kwargs, 'ylim', None)
        plot_robot = Utility.get_kwargs(kwargs, 'plot_robot', True)
        subplots = Utility.get_kwargs(kwargs, 'subplots', None)
        label = Utility.get_kwargs(kwargs, 'label', '')
        plot_text = Utility.get_kwargs(kwargs, 'plot_text', False)
        plot_target_terrain = Utility.get_kwargs(kwargs, 'plot_target_terrain', True) 
        xo, yo = structure_info.outline
        robot_position = robot_info.position
        robot_orientation = robot_info.orientation
        arrow_length = 50
        xr, yr = robot_position
        dxr, dyr = Vec2d(arrow_length, 0).rotated(robot_orientation)
        
        if subplots == None: 
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots
        fig.dpi = 200
        ax.plot(xo, yo, label=label)
        ax.plot(xo, xo*0, color='black')
        if plot_robot: 
            ax.scatter(xr, yr, c='red')
            ax.arrow(xr, yr, dxr, dyr, width=0.1, fc='red', ec='red')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title("time: " + str(np.round(simulation_info.simTime, 2)) + 's')
        ax.axis('equal')
        if xlim != None: 
            ax.set_xlim(xlim)
        if ylim != None: 
            ax.set_ylim(ylim)
        if plot_text: 
            text = "(pwmR, pwmL): " + str(tuple(robot_info.pwmRL))
            ax.text(0, 1, text, ha='left', va='top', transform=ax.transAxes) 
        if plot_target_terrain: 
            xt, yt = self.equalize_outline_integral((xo, yo), self.target_terrain_outline)
            ax.plot(xt, yt, '--', label='target structure', alpha=0.5)
        
        return fig, ax
    
    
    def plot_construction_comparison(self, construction_data, ind1=0, ind2=-1): 
        """
        By default compare initial and final construction  
        """
        # plot first construction instance
        if ind1 == 0: 
            label1 = 'initial structure'
        else: 
            simTime_str = str(np.round(construction_data.simulation_info_list[ind1].simTime, 2)) + 's'
            label1 = 'structure at t=' + simTime_str
        subplots = self.plot_construction(
            construction_data.structure_info_list[ind1], 
            construction_data.robot_info_list[ind1], 
            construction_data.simulation_info_list[ind1], 
            plot_robot=False, 
            label=label1, 
            plot_target_terrain=False,
            )
        # plot second construction instance 
        if ind2 == -1: 
            label2 = 'final structure' 
        else: 
            simTime_str = str(np.round(construction_data.simulation_info_list[ind2].simTime, 2)) + 's'
            label2 = 'structure at t=' + simTime_str 
        fig, ax = self.plot_construction(
            construction_data.structure_info_list[ind2], 
            construction_data.robot_info_list[ind2], 
            construction_data.simulation_info_list[ind2], 
            plot_robot=False, 
            label=label2, 
            subplots=subplots, 
            plot_target_terrain=True, 
            )
        ax.legend()
        ax.set_title(None)
        
        return fig, ax
    
    
    def plot_initial_and_target_terrain(self): 
        xi, yi = self.initial_terrain_info.goal_terrain # get initial terrain
        xt, yt = self.target_terrain_goal_terrain # get equalized target terrain 
        fig, ax = plt.subplots()
        fig.dpi = 200 
        ax.plot(xi, yi, label='initial terrain')
        ax.plot(xt, yt, label='target terrain')
        ax.plot(xi, xi*0) 
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        
        return fig, ax 
    
    
    def plot_grain_particles(self): 
        fig, ax = self.grains.plot_grain_particles() 
        xwall = [0, self.space_w, self.space_w, 0, 0]
        ywall = [0, 0, self.space_h, self.space_h, 0]
        ax.plot(xwall, ywall)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.axis('equal')
        
        return fig, ax 
        
        
    def film_construction(
            self, 
            construction_data, 
            save_loc='video/', 
            video_name='default', 
            image_loc='video/video_image/', 
            fps=50, 
            sampling_period=1, 
            ):
        # sample construction data 
        if sampling_period > 1: 
            construction_data_sampled = Construction_Data(
                Utility.sample_list(construction_data.structure_info_list, sampling_period), 
                Utility.sample_list(construction_data.robot_info_list, sampling_period), 
                Utility.sample_list(construction_data.simulation_info_list, sampling_period), 
                )
        else:
            construction_data_sampled = construction_data
        # make film
        with plt.ioff(): 
            for ii in tqdm(range(len(construction_data_sampled.structure_info_list))): 
                structure_info = construction_data_sampled.structure_info_list[ii]
                robot_info = construction_data_sampled.robot_info_list[ii]
                simulation_info = construction_data_sampled.simulation_info_list[ii]
                if ii == 0: 
                    fig, ax = self.plot_construction(structure_info, robot_info, simulation_info, 
                                                     plot_text=True)
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                else:
                    self.plot_construction(structure_info, robot_info, simulation_info, 
                                           plot_text=True, xlim=xlim, ylim=ylim)
                file_name = image_loc + 'img_' + str(ii) + '.png'
                plt.savefig(file_name, dpi=100)
                plt.close()
        save_loc_saved, video_name_saved = Utility.make_video(
            image_loc=image_loc, 
            save_loc=save_loc, 
            file_prefix='img', 
            file_extension='.png', 
            fps=fps, 
            video_name=video_name, 
            delete_images_afterwards=True,
            )
        print("Video is saved in: " + save_loc_saved + video_name_saved)
        
        return save_loc_saved, video_name_saved
    
    
    def film_pygame_video(
            self, 
            save_loc='video/', 
            video_name='default', 
            image_loc='img/simulation_screenshot/', 
            fps=50, 
            ):
        save_loc_saved, video_name_saved = Utility.make_video(
            image_loc=image_loc, 
            save_loc=save_loc, 
            file_prefix='img', 
            file_extension='.png', 
            fps=fps, 
            video_name=video_name, 
            delete_images_afterwards=True,
            )
        print("Video is saved in: " + save_loc_saved + video_name_saved)
        
        return save_loc_saved, video_name_saved
        
        
    def robot_collides_with_wall(self): 
        wci = Wall_Collision_Info(False, False) 
        if self.site_is_prepared: 
            collision = False 
            shape_r_list = self.robot.link_right_list[-1].shape_list + self.robot.link_left_list[-1].shape_list
            for ii, shape_w in enumerate([self.wall.shape_list[1], self.wall.shape_list[2]]): 
                for shape_r in shape_r_list: 
                    if detect_collision(shape_w, shape_r): 
                        collision = True 
                        if ii == 0: 
                            wci = Wall_Collision_Info(True, False)
                        elif ii == 1: 
                            wci = Wall_Collision_Info(False, True) 
                        break 
                if collision: 
                    break 
        
        return wci 
    
    
    def report_robot_motor_force(self): 
        mfRL = self.robot.report_motor_force(self.robot.dt / self.step_num) 
        
        return mfRL 
    
    
    def report_grain_pose(self): 
        grain_pose_info_list = self.grains.report_pose_all()
        
        return grain_pose_info_list 
    
    
    def report_grain_velocity(self): 
        gv_list = self.grains.report_velocity_all()
        
        return gv_list 
    
    
    def plot_robot_action(self, construction_data): 
        pwmR_list = []
        pwmL_list = []
        for robot_info in construction_data.robot_info_list: 
            pwmR_list.append(robot_info.pwmRL[0])
            pwmL_list.append(robot_info.pwmRL[1])
        fig, axs = plt.subplots(2, 1) 
        fig.set_size_inches(fig.get_size_inches()*np.array([1,1]))
        fig.dpi = 200 
        data_list = [pwmR_list, pwmL_list]
        title_list = ['pwmR', 'pwmL'] 
        for ii, data in enumerate(data_list): 
            ax = axs[ii]
            ax.plot(data)
            ax.set_xlabel('time step') 
            ax.set_title(title_list[ii])
        
        return fig, axs 
        
        
    def plot_robot_pose(self, construction_data): 
        xr_list = []
        yr_list = []
        thetar_list = []
        for robot_info in construction_data.robot_info_list: 
            xr_list.append(robot_info.position.x)
            yr_list.append(robot_info.position.y)
            thetar_list.append(robot_info.orientation)
        title_list = ['x position', 'y position', 'orientation']
        fig, axs = plt.subplots(3, 1) 
        fig.set_size_inches(fig.get_size_inches()*np.array([1,2]))
        fig.dpi = 200 
        for ii, data in enumerate([xr_list, yr_list, thetar_list]): 
            ax = axs[ii]
            ax.plot(data)
            ax.set_xlabel('time step') 
            ax.set_title(title_list[ii]) 
            
        return fig, axs 
    
    
    def characterize_construction(self, construction_data, **kwargs): 
        tag = Utility.get_kwargs(kwargs, 'tag', 'unnamed') 
        # plot structure 
        fig, ax = self.plot_construction_comparison(construction_data)
        ax.set_title(tag)
        # plot W1 distance 
        W1_all = np.array(self.get_W1_distance_all(construction_data)) 
        reduced_W1_all = W1_all[0] - W1_all 
        fig, ax = plt.subplots()
        fig.dpi = 200 
        ax.plot(reduced_W1_all)
        ax.plot(np.ones(reduced_W1_all.shape) * W1_all[0]) 
        ax.set_xlabel('time step')
        ax.legend(['reduced W1 distance', 'initial W1 distance'])
        ax.set_title(tag)
        # plot robot action 
        fig, axs = self.plot_robot_action(construction_data)
        fig.suptitle(tag)
        # plot robot pose 
        fig, axs = self.plot_robot_pose(construction_data)
        fig.suptitle(tag)
    
    
    def plot_simulation_error(self): 
        if self.simulation_error_analysis: 
            # gather data 
            diff_ThetaRL_matrix = np.degrees(np.vstack(self.diff_ThetaRL_list))
            m, n = diff_ThetaRL_matrix.shape
            colName = []
            for ii in range(int(n/2)): 
                colName.append('R' + str(ii))
            for ii in range(int(n/2)): 
                colName.append('L' + str(ii))
            # absolute link angle 
            df0 = pd.DataFrame(data=diff_ThetaRL_matrix, columns=colName)
            # relative link angle 
            diff_thetaRL_matrix = np.degrees(np.vstack(self.diff_thetaRL_list))
            df1 = pd.DataFrame(data=diff_thetaRL_matrix, columns=colName)
            # compensation motor rate 
            cmrRL_matrix = np.degrees(np.vstack(self.cmrRL_list))
            df2 = pd.DataFrame(data=cmrRL_matrix, columns=colName)
            # motor rate 
            mrRL_matrix = np.degrees(np.vstack(self.mrRL_list))
            df3 = pd.DataFrame(data=mrRL_matrix, columns=colName)
            
            # plot error in abs link angle 
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df0[cn])
            ax.set_title("error in abs angle over time (degree)")
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df0[cn].expanding().mean())
            ax.set_title("average error in abs angle over time (degree)")
            fig, ax = plt.subplots()
            fig.dpi=200
            ax.plot(df0.mean(), '*')
            ax.set_title("avg error in abs angle (degree)")
        
            # plot compensation motor rate 
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df2[cn] * -self.robot.dt)
            ax.set_title("compensation motor rate x -dt (degree)")
            
            # plot motor rate 
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df3[cn] * -self.robot.dt)
            ax.set_title("motor rate x -dt (degree)")
        
            # compare compensation motor rate and error in relative link angle 
            # small difference is expected 
            df_io_diff = df1.shift(periods=1, fill_value=0) - (df2 * -self.robot.dt)
            io_diff = df_io_diff.to_numpy()
            print("I/O difference for error correction:", np.sum(np.abs(io_diff)))
            # for cn in colName: 
            #     fig, ax = plt.subplots()
            #     fig.dpi=200
            #     # ax.plot(df0[cn], label="error in abs angle (degree)")
            #     ax.plot(df1[cn].shift(periods=1, fill_value=0), label="error in relative angle (degree)")
            #     ax.plot(df2[cn] * -self.robot.dt, label="compensation motor rate x -dt (degree)")
            #     # ax.plot(df1[cn].shift(periods=1, fill_value=0) - (df2[cn] * -robot.dt), label="input - output")
            #     ax.set_title(cn)
            #     ax.legend()
            #     ax.set_xlim([0,200])
            
            # plot error in relative link angle 
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df1[cn])
            ax.set_title("error in relative angle over time (degree)")
            fig, ax = plt.subplots()
            fig.dpi=200
            for cn in colName: 
                ax.plot(df1[cn].expanding().mean())
            ax.set_title("average error in relative angle over time (degree)")
            fig, ax = plt.subplots()
            fig.dpi=200
            ax.plot(df1.mean(), '*')
            ax.set_title("avg error in relative angle (degree)")
        
        
# In[]
if __name__ == "__main__": 
    run = 0 
    if run: 
        initial_terrain_info = Utility.load_data('data/terrain/', 'incline_poly_r7e5_k0.2_w1500mm.pkl')
        target_terrain_info = Utility.load_data('data/terrain/', 'ground_poly_r7e5_h200mm_w1500mm.pkl')
    run = 0 
    if run: 
        initial_terrain_info = Utility.load_data('data/terrain/', 'incline_poly_r7e5_k0.2_w1000mm.pkl')
        target_terrain_info = Utility.load_data('data/terrain/', 'ground_poly_r7e5_h200mm_w1000mm.pkl')
    run = 1 
    if run: 
        initial_terrain_info = Utility.load_data('data/terrain/', 'incline_poly_r7e5_k0.2_w750mm.pkl')
        target_terrain_info = Utility.load_data('data/terrain/', 'ground_poly_r7e5_h200mm_w750mm.pkl')
    run = 0 
    if run: 
        initial_terrain_info = Utility.load_data('data/terrain/', 'floor_h500mm_w1500mm.pkl')
        target_terrain_info = Utility.load_data('data/terrain/', 'floor_h500mm_w1500mm.pkl')
        
    data_autosave = 0
    simulator = Simulator(
        initial_terrain_info=initial_terrain_info, 
        target_terrain_info=target_terrain_info, 
        pygame_visualization=1, 
        data_autosave=data_autosave, 
        )
    simulator.plot_initial_and_target_terrain()
    simulator.prepare_site(x_robot_drop=750)
    pwmR, pwmL = (255, -191)
    pwmR, pwmL = (255, 255)
    pwmR, pwmL = (0, 0)
    pwmR, pwmL = (255, 95)
    simulator.update_structure_info = 1
    for ii in tqdm(range(20)): 
        simulator.construct(pwmR, pwmL)
    simulator.stop_simulation()
    
    if data_autosave: 
        cd = Utility.load_data(*simulator.saved_construction_data_dir)
        simulator.plot_W1_distance(cd) 
        simulator.plot_construction_comparison(cd) 
        
        
        
        











