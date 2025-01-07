# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:32:14 2023

@author: ericl
"""


# In[]
# import pygame
import pymunk
import pymunk.autogeometry
import pymunk.pygame_util
from pymunk import Vec2d

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# import pickle

import Utility


# In[IMPORTANT NOTES]


# In[]
red = (255, 0, 0, 0)
red2 = (128, 0, 0, 0)
green = (0, 255, 0, 0)
blue = (0, 0, 255, 0)
blue2 = (0, 0, 128, 0)
yellow = (255, 255, 0, 0)

BoundingBox = namedtuple('BoundingBox', ['top', 'bottom', 'left', 'right'])
Grain_Info = namedtuple('Grain_Info', [
    'grain_type', 
    'density', 
    'friction', 
    'elasticity', 
    'radius', 
    'edge_num', 
    'scale', 
    'sectionNum', 
    ])
Terrain_Info = namedtuple('Terrain_Info', [
    'grain_pos_list', 
    'grain_angle_list', 
    'grain_info', 
    'goal_terrain', 
    'space_size', 
    ])
Grain_Pose_Info = namedtuple('Grain_Pose_Info', [
    'initial_position', 
    'position', 
    'initial_orientation', 
    'orientation', 
    ])

robot_density = 1

robot_ShapeFilter_GroupNumber = 1
wall_ShapeFilter_GroupNumber = 2
robot_ShapeFilter = pymunk.ShapeFilter(group=robot_ShapeFilter_GroupNumber)
wall_ShapeFilter = pymunk.ShapeFilter(group=wall_ShapeFilter_GroupNumber)


# In[]
def get_limit_velocity_func(max_velocity): 
    '''
    Generate a callback function for limiting the velocity 
    CAUTION: using this function may deteriorate simulation efficiency 
    '''
    def limit_velocity_func(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        l = body.velocity.length
        if l > max_velocity:
            scale = max_velocity / l
            body.velocity = body.velocity * scale
            
    return limit_velocity_func


# In[]
class PivotJoint:
    def __init__(self, space, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.constraints.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        
        space.add(joint)
        
        
class DampedRotarySpring:
    def __init__(self, space, b, b2, angle=0, stiffness=1e+9, damping=1e+6):
        joint = pymunk.constraints.DampedRotarySpring(b, b2, angle, stiffness, damping)
        
        space.add(joint)
        
        
class Motor: 
    def __init__(self, space, b, b2, rate=0, max_force=1e+11, tag=None):
        self.tag = tag
        self.motor = pymunk.constraints.SimpleMotor(b, b2, rate)
        self.motor.max_force = max_force 
        
        space.add(self.motor)
        
    def report_rate(self): 
        return self.motor.rate 
        
    def update_rate(self, rate): 
        self.motor.rate = rate
        
    def modify_rate(self, drate): 
        self.motor.rate += drate
        
    def deactivate(self): 
        self.motor.rate = 0
            
            
class Link:
    def __init__(self, space, pos, v, r, thr, thl):
        self.body = pymunk.Body()
        self.body.position = pos
        self.pos_init = pos
        self.v = v
        
        # add shape
        pt = Vec2d(v.x/2, v.y/2)
        vt = v.perpendicular_normal() * r
        self.shape_list = [
            pymunk.Segment(self.body, (0, 0), v, thl/2), # link
            pymunk.Segment(self.body, pt - vt, pt + vt, thr/2), # tip 
            ]
        
        # set up physical parameters
        for shape in self.shape_list: 
            shape.friction = 1
            shape.density = robot_density
            shape.elasticity = 0.0
            shape.color = green
            shape.filter = robot_ShapeFilter
        
        space.add(self.body, *self.shape_list)
        
        
class Linkf: 
    """
    final link with tail cap 
    """
    def __init__(self, space, pos, v, r, shape_type=3): 
        self.body = pymunk.Body()
        self.body.position = pos
        self.pos_init = pos
        self.v = v
        th = 2
        
        # add tail cap
        p0 = Vec2d(0, 0)
        vt = v.perpendicular_normal() * r
        if Utility.rad2pipi(vt.angle) < 0: 
            vt = vt*-1
        if shape_type == 1: # triangle 
            self.shape_list = [
                pymunk.Segment(self.body, vt, -vt, th/2), 
                pymunk.Segment(self.body, vt, v, th/2), 
                pymunk.Segment(self.body, -vt, v, th/2), 
                ]
        elif shape_type == 2: # scoop
            self.shape_list = [
                pymunk.Segment(self.body, vt, -vt, th/2), 
                pymunk.Segment(self.body, p0, v, th/2), 
                ]
        elif shape_type == 3: # triangle scoop
            s = 2/3
            self.shape_list = [
                pymunk.Segment(self.body, vt, -vt, th/2), 
                pymunk.Segment(self.body, vt, (v+vt)*s-vt, th/2), 
                pymunk.Segment(self.body, v, (v+vt)*s-vt, th/2), 
                ]
        
        # set up physical parameters
        for shape in self.shape_list: 
            shape.friction = 1
            shape.density = robot_density
            shape.elasticity = 0.0
            shape.color = green 
            shape.filter = robot_ShapeFilter
            
        space.add(self.body, *self.shape_list)
        
        
class Chassis:
    def __init__(self, space, pos, h, w):
        self.body = pymunk.Body()
        self.body.position = pos
        self.vs_list = [
            Vec2d(-w/2, -h/2), 
            Vec2d(-w/2, h/2), 
            Vec2d(w/2, -h/2), 
            Vec2d(w/2, h/2), 
            ]
        self.pos_init = pos

        self.shape = pymunk.Poly(self.body, self.vs_list)
        
        # set up physical parameters
        for shape in [self.shape]: 
            shape.friction = 1
            shape.filter = robot_ShapeFilter
            shape.density = robot_density
            shape.elasticity = 0.0
            shape.color = yellow 
        
        space.add(self.body, self.shape)
        
        
class EndEffector: 
    def __init__(self, space, pos, add_shape=True): 
        self.body = pymunk.Body()
        self.body.position = pos
        self.pos_init = pos
        
        if add_shape: 
            self.shape = pymunk.Circle(self.body, 1)
            self.shape.filter = robot_ShapeFilter
            self.shape.density = 0.01
            self.shape.elasticity = 0.0
            space.add(self.body, self.shape)
        else:
            space.add(self.body)
        
        
class Circle:
    def __init__(self, space, pos, radius):
        self.body = pymunk.Body()
        self.body.position = pos
        self.pos_init = pos
        self.radius = radius
        self.diameter = self.radius * 2
        
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 0.25
        self.shape.friction = 100
        self.shape.elasticity = 0.0
        
        space.add(self.body, self.shape)
        
        
# In[]
class Wall:
    def __init__(self, space, h, w, thickness=10): 
        self.h = h
        self.w = w
        self.r = thickness/2
        
        self.shape_list = [
            pymunk.Segment(space.static_body, (0, 0-self.r), (self.w, 0-self.r), self.r), # floor
            pymunk.Segment(space.static_body, (0-self.r, 0), (0-self.r, self.h), self.r), # left wall
            pymunk.Segment(space.static_body, (self.w+self.r, 0), (self.w+self.r, self.h), self.r), # right wall 
            ]
        for shape in self.shape_list: 
            shape.friction = 1
            shape.elasticity = 0
            shape.filter = wall_ShapeFilter
            
        space.add(*self.shape_list)
        
        
    def add_lid(self, space): 
        lid = pymunk.Segment(space.static_body, (0, self.h+self.r), (self.w, self.h+self.r), self.r)
        lid.friction = 1
        lid.elasticity = 0
        lid.filter = wall_ShapeFilter
        self.shape_list.append(lid)
        
        space.add(lid)
        
        
class Wall_for_Test:
    def __init__(self, space, h, w, thickness=10): 
        self.h = h
        self.w = w
        self.r = thickness/2
        
        self.shape_list = [
            pymunk.Segment(space.static_body, (0, 0-self.r), (self.w, 0-self.r), self.r), # floor
            pymunk.Segment(space.static_body, (0-self.r, 0), (0-self.r, self.h), self.r), # left wall
            pymunk.Segment(space.static_body, (self.w+self.r, 0), (self.w+self.r, self.h), self.r), # right wall
            ]
        for shape in self.shape_list: 
            shape.friction = 1
            shape.elasticity = 0
            shape.filter = wall_ShapeFilter
            
        space.add(*self.shape_list)
        
        
    def remove_terrain(self, space): 
        """
        remove the last added terrain 
        """
        if len(self.shape_list) > 0: 
            space.remove(self.shape_list[-1])
            self.shape_list.pop()
        
        
    def add_partition(self, space, wp): 
        """
        For finding the maximum steepness of the granular material
        
        left wall      partition   right wall
        |                  |           |
        |                  |           |
        |                  |           |
        |__________________|___________|
                           wp          w
        """
        shape = pymunk.Segment(space.static_body, (wp+self.r, 0), (wp+self.r, self.h), self.r)
        shape.friction = 1
        shape.elasticity = 0
        shape.filter = wall_ShapeFilter
        self.shape_list.append(shape)
        space.add(shape)
        
        
    def add_incline(self, space, ws, k): 
        """
        For finding the maximum steepness of the granular material
        
        left wall                    right wall
        |                      *       |
        |                   *          |
        |                *             |
        |_____________*________________|
                     ws                w
        """
        vs_list = [(ws, 0), (self.w, 0), (self.w, (self.w-ws)*k)]
        shape = pymunk.Poly(space.static_body, vs_list)
        shape.friction = 1
        shape.elasticity = 0
        shape.filter = wall_ShapeFilter
        self.shape_list.append(shape)
        space.add(shape)
        
        
    def add_mount(self, space, ws, k, wm, hm): 
        """
        For finding the maximum steepness of the granular material
        
        left wall                             right wall
        |                    ______________ hm     |
        |                   *              *       |
        |                *                    *    |
        |_____________*__________________________*_|
                     ws        <-  wm  ->          w
        """
        x1 = hm/k + ws
        x2 = x1 + wm
        x3 = x2 + hm/k
        vs_list = [(ws, 0), (x1, hm), (x2, hm), (x3, 0)]
        shape = pymunk.Poly(space.static_body, vs_list)
        shape.friction = 1
        shape.elasticity = 0
        shape.filter = wall_ShapeFilter
        self.shape_list.append(shape)
        space.add(shape)
        
        
    def add_tube(self, space, ht, wt, hd): 
        p1 = Vec2d(self.w/2 - wt/2, hd)
        p2 = Vec2d(self.w/2 + wt/2, p1.y)
        p3 = Vec2d(p1.x, hd + ht)
        p4 = Vec2d(p2.x, p3.y)
        offset_x = Vec2d(self.r, 0)
        offset_y = Vec2d(0, self.r)
        shape_list = [
            pymunk.Segment(space.static_body, p1-offset_x, p3-offset_x, self.r), # left wall
            pymunk.Segment(space.static_body, p2+offset_x, p4+offset_x, self.r), # right wall
            pymunk.Segment(space.static_body, p1-offset_y, p2-offset_y, self.r), # bottom lid
            ]
        for shape in shape_list: 
            shape.friction = 0
            shape.elasticity = 0
            shape.filter = wall_ShapeFilter
        self.shape_list += shape_list
        space.add(*shape_list)
        
        return p1, p2, p3, p4
    
    
# In[]
class Grain_Ball_Shape: 
    def __init__(self, 
                 pos, 
                 radius=5, 
                 density=0.1, 
                 friction=100, 
                 elasticity=0, 
                 ): 
        self.pos_init = pos
        self.angle_init = 0
        self.radius = radius
        self.diameter = self.radius * 2
        self.density = density
        self.friction = friction
        self.elasticity = elasticity
        self.type = 'Grain_Ball_Shape'
        self.edge_num = 1
        
        
    def _get_body(self, space, position, body_type, max_velocity=1e+4): 
        """
        This function should only be called by self.add
        """
        self.pos_init = position
        body = pymunk.Body(body_type=body_type)
        body.position = position
        
        # limit velocity 
        # this function is disabled due to the deterioration of simulation efficiency 
        # body.velocity_func = get_limit_velocity_func(max_velocity) 
        
        return body
        
        
    def _get_shape(self, body, shape_color, angle): 
        """
        This function should only be called by self.add
        """
        self.angle_init = angle
        shape = pymunk.Circle(body, self.radius)
        shape.density = self.density
        shape.friction = self.friction
        shape.elasticity = self.elasticity
        shape.color = shape_color
        shape_list = [shape]
        
        return shape_list
        
        
    def add(self, space, **kwargs): 
        body_type = Utility.get_kwargs(kwargs, 'body_type', pymunk.Body.DYNAMIC)
        position = Utility.get_kwargs(kwargs, 'position', self.pos_init)
        shape_color = Utility.get_kwargs(kwargs, 'shape_color', blue)
        angle = Utility.get_kwargs(kwargs, 'angle', self.angle_init)
        body = self._get_body(space, position, body_type)
        shape_list = self._get_shape(body, shape_color, angle)
        space.add(body, *shape_list)
        self.body = body
        self.shape_list = shape_list
        
        
    def remove(self, space): 
        space.remove(self.body, *self.shape_list)
        self.body = None
        self.shape_list = []
        
        
    def set_static(self, space): 
        # record current grain spec and remove it and then add new grain 
        if self.body.body_type == pymunk.Body.DYNAMIC: 
            position = self.body.position
            angle = Utility.rad2pipi(self.body.angle + self.angle_init)
            shape_color = self.shape_list[0].color
            self.remove(space)
            self.add(space, body_type=pymunk.Body.STATIC, 
                     position=position, shape_color=shape_color, angle=angle)
        
        
    def set_dynamic(self): 
        if self.body.body_type == pymunk.Body.STATIC: 
            self.body.body_type = pymunk.Body.DYNAMIC
            
            
    def report_pose(self): 
        grain_pose_info = Grain_Pose_Info(self.pos_init, self.body.position, self.angle_init, self.body.angle)
        
        return grain_pose_info
        
        
    def report_area(self): 
        area = 0
        for shape in self.shape_list: 
            area += shape.area 
            
        return area 
        
        
class Grain_Poly_Shape(Grain_Ball_Shape): 
    """
    convex polygon
    """
    def __init__(self, 
                 pos, 
                 radius=5, 
                 edge_num=3, 
                 density=0.1, 
                 friction=100, 
                 elasticity=0, 
                 irregular=False, 
                 ): 
        super().__init__(pos, radius, density, friction, elasticity)
        self.type = 'Grain_Poly_Shape'
        self.edge_num = edge_num
        # get all vertices of a regular convex polygon 
        vs_list = [Vec2d(radius, 0)]
        rotation_angle = np.radians(360 / edge_num)
        for ii in range(edge_num-1): 
            vs_last = vs_list[-1]
            vs_new = vs_last.rotated(rotation_angle)
            vs_list.append(vs_new)
        # randomize vertices 
        if not irregular: 
            self.vs_list = vs_list
        else:
            self.type = 'Grain_PolyIrregular_Shape'
            randomization_scale = 3/4
            variation_angle = rotation_angle / 2 * randomization_scale
            self.vs_list = []
            for vs in vs_list: 
                random_angle = random.uniform(-variation_angle, variation_angle)
                vsr = vs.rotated(random_angle)
                self.vs_list.append(vsr)
            
            
    def _get_shape(self, body, shape_color, angle): 
        """
        This function should only be called by self.add
        """
        self.angle_init = angle
        new_vs_list = []
        for vs in self.vs_list: 
            new_vs_list.append(vs.rotated(angle))
        shape = pymunk.Poly(body, new_vs_list)
        shape.density = self.density
        shape.friction = self.friction
        shape.elasticity = self.elasticity
        shape.color = shape_color
        shape_list = [shape]
        
        return shape_list
    
    
# In[]
class Grains: 
    def __init__(
            self, 
            container_height, 
            container_width, 
            grain_density=0.1, 
            grain_friction=100, 
            grain_elasticity=0, 
            random_seed=None, 
            ): 
        self.container_height = container_height
        self.container_width = container_width
        self.random_seed = random_seed
        random.seed(random_seed)  # random seed by default
        
        # grain parameters
        self.grain_density = grain_density
        self.grain_friction = grain_friction
        self.grain_elasticity = grain_elasticity
        self.grain_list = []
        self.grain_num = 0
        self.grain_added = False
        
        # other parameters
        self.last_robot_position = None # used for control sleep_selectively()
        self.last_robot_orientation_vector = None # used for control sleep_selectively() 
        self.grain_outside_container_ind_list = [] # index of grains that get out of the container 
        
        # error messages 
        self.warning_msg_list = [] # list of warning messages 
        self.fatal_error_msg_list = [] # list of fatal error messages 
        
        
    def add_one_grain(self, space, pos, **kwargs): 
        angle = Utility.get_kwargs(kwargs, 'angle', 0)
        grain_type = Utility.get_kwargs(kwargs, 'grain_type', "Grain_Ball_Shape")
        radius = Utility.get_kwargs(kwargs, 'radius', 7)
        edge_num = Utility.get_kwargs(kwargs, 'edge_num', 5)
        
        if grain_type == "Grain_Ball_Shape": 
            grain = Grain_Ball_Shape(pos, radius, 
                                     self.grain_density, self.grain_friction, self.grain_elasticity)
        elif grain_type == "Grain_Poly_Shape": 
            grain = Grain_Poly_Shape(pos, radius, edge_num, 
                                     self.grain_density, self.grain_friction, self.grain_elasticity)
        elif grain_type == "Grain_PolyIrregular_Shape": 
            grain = Grain_Poly_Shape(pos, radius, edge_num, 
                                     self.grain_density, self.grain_friction, self.grain_elasticity, 
                                     irregular=True)
            
        grain.add(space, angle=angle)
        self.grain_list.append(grain)
        self.grain_num += 1
        
        
    def add_grains_random(self, space, total_grain_num, grain_type="Grain_Ball_Shape", **kwargs): 
        if total_grain_num > 0: 
            # limit the area of dropping materials 
            xlim = Utility.get_kwargs(kwargs, 'xlim', (0, self.container_width))
            ylim = Utility.get_kwargs(kwargs, 'ylim', (0, self.container_height))
            radius = Utility.get_kwargs(kwargs, 'radius', 7)
            edge_num = Utility.get_kwargs(kwargs, 'edge_num', 5)
            
            if grain_type == "Grain_Ball_Shape": 
                for ii in range(total_grain_num): 
                    x = random.randint(int(xlim[0]+radius)+1, int(xlim[1]-radius)-1)
                    y = random.randint(int(ylim[0]+radius)+1, int(ylim[1]-radius)-1)
                    self.add_one_grain(space, Vec2d(x, y), grain_type=grain_type, 
                                       radius=radius)
                    
            elif grain_type in ["Grain_Poly_Shape", "Grain_PolyIrregular_Shape"]: 
                for ii in range(total_grain_num): 
                    x = random.randint(int(xlim[0]+radius)+1, int(xlim[1]-radius)-1)
                    y = random.randint(int(ylim[0]+radius)+1, int(ylim[1]-radius)-1)
                    self.add_one_grain(space, Vec2d(x, y), grain_type=grain_type, 
                                       radius=radius, edge_num=edge_num)
            
        self.grain_added = True
        
    
    def add_flat_ground(self, space, height, grain_type="Grain_Ball_Shape", **kwargs): 
        """
        This function currently only supports ball and polygon shape

        Parameters
        ----------
        space : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.
        grain_type : TYPE, optional
            DESCRIPTION. The default is "Grain_Ball_Shape".
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        edge_num = Utility.get_kwargs(kwargs, 'edge_num', 5)
        radius = Utility.get_kwargs(kwargs, 'radius', 7)
            
        if height > radius*2: 
            layer_num = int(((height/radius) - 2 + 3**0.5) / 3**0.5)
            for ll in range(layer_num): 
                max_nbpl = int((self.container_width - ll%2) // (radius*2)) # max number of balls per layer
                for ii in range(max_nbpl): 
                    x = (ll%2 + 1)*radius + ii*2*radius
                    y = radius + ll * 3**0.5 * radius
                    if grain_type == "Grain_Ball_Shape": 
                        self.add_one_grain(space, Vec2d(x, y), grain_type=grain_type, 
                                           radius=radius)
                    elif grain_type in ["Grain_Poly_Shape", "Grain_PolyIrregular_Shape"]: 
                        self.add_one_grain(space, Vec2d(x, y), grain_type=grain_type, 
                                           radius=radius, edge_num=edge_num)
                    
        self.grain_added = True
        
        
    def save_terrain_info(self, file_name, **kwargs): 
        plot = Utility.get_kwargs(kwargs, 'plot', False)
        xg = []
        yg = []
        grain_pos_list = []
        grain_angle_list = []
        for grain in self.grain_list: 
            pos = grain.body.position
            grain_pos_list.append(pos)
            xg.append(pos.x)
            yg.append(pos.y)
            grain_angle = Utility.rad2pipi(grain.angle_init + grain.body.angle)
            grain_angle_list.append(grain_angle)
        if plot: 
            fig, ax = plt.subplots()
            fig.dpi = 200
            ax.plot(xg, yg, '*')
            ax.set_title('saved terrain')
        if grain.type in ["Grain_Ball_Shape", "Grain_Poly_Shape"]: 
            grain_info = Grain_Info(grain.type, grain.density, grain.friction, grain.elasticity, 
                                    grain.radius, grain.edge_num, None, None)
        xt = Utility.get_kwargs(kwargs, 'xt')
        yt = Utility.get_kwargs(kwargs, 'yt')
        space_h = Utility.get_kwargs(kwargs, 'space_h')
        space_w = Utility.get_kwargs(kwargs, 'space_w')
        save_loc = Utility.get_kwargs(kwargs, 'save_loc', 'data/terrain/')
        goal_terrain = (xt, yt)
        space_size = (space_h, space_w)
        terrain_info = Terrain_Info(grain_pos_list, grain_angle_list, grain_info, goal_terrain, space_size)
        Utility.save_data(terrain_info, save_loc=save_loc, file_name=file_name)
        
        return terrain_info
        
        
    def form_terrain(self, space, terrain_info, **kwargs):
        x_offset = Utility.get_kwargs(kwargs, 'x_offset', 0)
        y_offset = Utility.get_kwargs(kwargs, 'y_offset', 0)
        grain_pos_list = terrain_info.grain_pos_list
        grain_angle_list = terrain_info.grain_angle_list
        grain_info = terrain_info.grain_info
        grain_type = grain_info.grain_type
        radius = grain_info.radius
        edge_num = grain_info.edge_num
        
        if len(grain_pos_list) > 0: 
            for ii in range(len(grain_pos_list)): 
                pos = grain_pos_list[ii]
                angle = grain_angle_list[ii]
                pos_offset = Vec2d(pos.x+x_offset, pos.y+y_offset)
                if grain_type == "Grain_Ball_Shape": 
                    self.add_one_grain(space, pos_offset, angle=angle, 
                                       grain_type=grain_type, radius=radius)
                elif grain_type == "Grain_Poly_Shape": 
                    self.add_one_grain(space, pos_offset, angle=angle, 
                                       grain_type=grain_type, radius=radius, edge_num=edge_num)   
                
        self.grain_added = True
        
        
    def is_static(self, relaxed_condition=True): 
        if len(self.grain_list) > 0: 
            bodies_are_static = True
            for grain in self.grain_list: 
                pass_test = True
                if relaxed_condition: 
                    if grain.body.velocity.length > 1: 
                        pass_test = False
                else:
                    if not grain.body.is_sleeping and grain.body.body_type != pymunk.Body.STATIC: 
                        pass_test = False
                if not pass_test: 
                    bodies_are_static = False
                    break
        else:
            bodies_are_static = True
            
        return bodies_are_static
    
    
    def sleep_all(self, space): 
        if len(self.grain_list) > 0: 
            for grain in self.grain_list: 
                grain.set_static(space)
                
                
    def sleep_selectively(self, space, robot, **kwargs): 
        """
        This function should not be called frequently, otherwise simulation instability may occur
        Currently this function defines a box-shape active region, outside which particles are static
        In every call, the function checks if the robot is out of the active region or too close to borders
        If the above test returns True then this function updates the active region based on current robot location
        
        Current dimension of the active region: 
            width: 3x robot width 
            height: 5x robot height
        
        To perform the above test: 
            Check if robot center position (xr, yr) descends by 1x robot height: 
                h_margin: 1.5x robot height (critical)
                w_margin: 0.5x robot width 
            Check if two ends of robot (xeeR, yeeR), (xeeL, yeeL) is 0.5x robot width away from the border: 
                h_margin: 0.5x robot height
                w_margin: 0.5x robot width (critical)
        """        
        if len(self.grain_list) > 0: 
            # robot parameters
            update_active_region = False
            robot_orientation_vector = robot.get_orientation_vector()
            robot_position = robot.get_position()
            eeR_position = robot.get_end_effector_position(side='right')
            eeL_position = robot.get_end_effector_position(side='left')
            xr = robot_position.x
            yr = robot_position.y
            xeeR = eeR_position.x
            yeeR = eeR_position.y
            xeeL = eeL_position.x
            yeeL = eeL_position.y
            
            # active region parameters
            scale = Utility.get_kwargs(kwargs, 'scale', 1)
            print_msg = Utility.get_kwargs(kwargs, 'print_msg', True)
            h = robot.height * 2.5 * scale
            w = robot.width * 1.5 * scale
            
            # check if robot is out of or too close to the border of the active region (except top border)
            if self.last_robot_position == None and self.last_robot_orientation_vector == None: 
                update_active_region = True 
            else:
                # check center position of the robot 
                theta = self.last_robot_orientation_vector.angle
                x0 = self.last_robot_position.x
                y0 = self.last_robot_position.y
                h_margin = 1.5 * robot.height
                w_margin = 0.5 * robot.width
                inside, CTB = Utility.point_inside_box(xr, yr, h, w, theta, x0, y0, h_margin, w_margin)
                if not inside or np.any(CTB[0:3]): 
                    update_active_region = True 
                else:
                    # check right end effector
                    h_margin = 0.5 * robot.height
                    w_margin = 0.5 * robot.width
                    inside, CTB = Utility.point_inside_box(xeeR, yeeR, h, w, theta, x0, y0, h_margin, w_margin)
                    if not inside or np.any(CTB[0:3]): 
                        update_active_region = True 
                    else:         
                        # check left end effector
                        h_margin = 0.5 * robot.height
                        w_margin = 0.5 * robot.width
                        inside, CTB = Utility.point_inside_box(xeeL, yeeL, h, w, theta, x0, y0, h_margin, w_margin)
                        if not inside or np.any(CTB[0:3]): 
                            update_active_region = True 
            
            # make grains inside the bounding box sleep
            if update_active_region: 
                self.last_robot_orientation_vector = robot_orientation_vector 
                self.last_robot_position = robot_position 
                if print_msg: print("Active region is updated!")
                for grain in self.grain_list: 
                    xg = grain.body.position.x 
                    yg = grain.body.position.y 
                        
                    # check if inside box
                    theta = robot_orientation_vector.angle
                    x0 = xr
                    y0 = yr
                    inside, CTB = Utility.point_inside_box(xg, yg, h, w, theta, x0, y0)
                    if inside: 
                        grain.set_dynamic()
                    else:
                        grain.set_static(space)
                
                
    def activate_all(self): 
        if len(self.grain_list) > 0: 
            for grain in self.grain_list: 
                grain.set_dynamic()
                
                
    def remove_one_grain(self, space, ind): 
        if len(self.grain_list) > 0: 
            grain = self.grain_list.pop(ind)
            grain.remove(space)
            self.grain_num -= 1
                
                
    def remove_all(self, space): 
        if len(self.grain_list) > 0: 
            for ii in range(len(self.grain_list)): 
                grain = self.grain_list.pop()
                grain.remove(space)
                self.grain_num -= 1
                
                
    def remove_grain_outside_container(self, space): 
        """
        Remove grains outside the container and add new ones randomly 
        Raise errors if too many grains get out 
        """
        grain_outside_num = len(self.grain_outside_container_ind_list)
        # if small number of grains get out of the container, remove them 
        if grain_outside_num > 0 and grain_outside_num <= 2:  
            self.grain_outside_container_ind_list.sort(reverse=True) 
            for grain_ind in self.grain_outside_container_ind_list: 
                self.remove_one_grain(space, grain_ind) 
            self.grain_outside_container_ind_list = []
            # add new grains 
            grain0 = self.grain_list[0]
            if grain0.type == 'Grain_Ball_Shape': 
                self.add_grains_random(space, grain_outside_num, grain_type=grain0.type, 
                                       radius=grain0.radius, 
                                       ylim=(self.container_height-grain0.radius*4, self.container_height)) 
            elif grain0.type in ['Grain_Poly_Shape', 'Grain_PolyIrregular_Shape']: 
                self.add_grains_random(space, grain_outside_num, grain_type=grain0.type, 
                                       radius=grain0.radius, edge_num=grain0.edge_num, 
                                       ylim=(self.container_height-grain0.radius*4, self.container_height)) 
            # add warning messages 
            warning_msg = 'Warning: ' + str(grain_outside_num) + ' grain(s) outside the container!'
            self.warning_msg_list.append(warning_msg) 
        # if large number of grains get out of the container, report fatal error 
        elif grain_outside_num > 2: 
            error_msg = 'Fatal Error: grain explosion!' 
            self.fatal_error_msg_list.append(error_msg)
                
                
    def shuffle_all(self, strength=1e+3, print_msg=False): 
        if len(self.grain_list) > 0: 
            imps = Vec2d(strength, 0)
            for grain in self.grain_list: 
                angle = random.uniform(-np.pi, np.pi)
                imp = imps.rotated(angle)
                grain.body.apply_impulse_at_local_point(imp)
            if print_msg: 
                print("Grains are shuffled!")
                
                
    def get_compactness(self, space): 
        compactness = 0
        if len(self.grain_list) > 0: 
            grain_area = 0
            for grain in self.grain_list: 
                grain_area_single = grain.report_area()
                grain_area += grain_area_single
            xo, yo = self.get_outline(space, False, False)
            structure_area = np.trapz(yo, x=xo)
            compactness = grain_area / structure_area
        
        return compactness
    
    
    def color_grain(self, grain, highlight=False): 
        for shape in grain.shape_list: 
            if highlight: 
                if grain.body.body_type == pymunk.Body.DYNAMIC: 
                    shape.color = red
                else:
                    shape.color = red2
            else: 
                if grain.body.body_type == pymunk.Body.DYNAMIC: 
                    shape.color = blue
                else:
                    shape.color = blue2
    
    
    def get_outline(self, space, steepness_max=1e+6, highlight=True, plot=False):
        """
        This function serves several purposes: 
            finds the outline of the granular material 
            detects grains outside the container 
            remove grains outside the container 
        
        Outline points satisfy following conditions: 
            No particles (shapes) within the upright 2-particle-size box area 
                these shapes belong to wall/robot
            No robot underneath 

        Parameters
        ----------
        space : TYPE
            DESCRIPTION.
        highlight : TYPE, optional
            DESCRIPTION. The default is True.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        x_outline : TYPE
            DESCRIPTION.
        y_outline : TYPE
            DESCRIPTION.

        """
        grain_outline_ind_list = []
        x_outline = np.arange(0, 1000, 10)
        y_outline = x_outline * 0
        if len(self.grain_list) > 0: 
            grain_sample = self.grain_list[0]
            
            if grain_sample.type in ["Grain_Ball_Shape", "Grain_Poly_Shape", "Grain_PolyIrregular_Shape"]: 
                for grain_ind, grain in enumerate(self.grain_list): 
                    # get particle dimension 
                    pos = grain.body.position
                    radius = grain.radius 
                    
                    # check if the particle is out of the container 
                    # log index of outside grain and remove them later 
                    grain_inside_container = True  
                    if pos.x < 0 or pos.x > self.container_width or pos.y < 0 or pos.y > self.container_height: 
                        grain_inside_container = False 
                        self.grain_outside_container_ind_list.append(grain_ind)
                    
                    # if grain is insdie the container, check if it is an outline particle 
                    if grain_inside_container: 
                        # check bounding box area above the particle 
                        # bottom of the box is 0.1 higher than the top of the particle, box alighs with the particle 
                        # width: width_ratio x 2r 
                        # height: 4r
                        # there should be no particles above the outline particle
                        clear_above = True
                        width_ratio = 1/4 # width_ratio = box_width / particle diameter
                        bb = pymunk.BB(left=pos.x-radius*width_ratio, 
                                       bottom=pos.y+radius+0.1, 
                                       right=pos.x+radius*width_ratio, 
                                       top=pos.y+radius*5+0.1)
                        shape_list = space.bb_query(bb, pymunk.ShapeFilter())
                        if len(shape_list) == 0: 
                            pass
                        else: 
                            for shape in shape_list: 
                                if shape.filter[0] not in [robot_ShapeFilter_GroupNumber, wall_ShapeFilter_GroupNumber]: 
                                    clear_above = False
                                    break
                                
                        # if area avove is clear, check bounding box area below the particle 
                        # region 1: 
                        #   top of the box is 1 below the bottom of the particle, box alighs with the particle 
                        #   width = width_ratio x 2r, height = r 
                        #   the particle should not be floating 
                        # region 2: 
                        #   top of the box is 1 below the bottom of the particle, box alighs with the particle, bottom is 0 
                        #   width = width_ratio x 2r
                        #   the particle should not be over the robot 
                        clear_all = True 
                        if clear_above: 
                            width_ratio = 1 # width_ratio = box_width / particle diameter
                            
                            # check region 1 
                            bb = pymunk.BB(left=pos.x-radius*width_ratio, 
                                           bottom=pos.y-2*radius-1, 
                                           right=pos.x+radius*width_ratio, 
                                           top=pos.y-radius-1)
                            shape_list = space.bb_query(bb, pymunk.ShapeFilter())
                            if len(shape_list) == 0:
                                clear_all = False # particle should not be floating 
                                
                            # check region 2
                            if clear_all: 
                                bb = pymunk.BB(left=pos.x-radius*width_ratio, 
                                               bottom=0, 
                                               right=pos.x+radius*width_ratio, 
                                               top=pos.y-radius-1)
                                shape_list = space.bb_query(bb, pymunk.ShapeFilter())
                                for shape in shape_list: 
                                    if shape.filter[0] == robot_ShapeFilter_GroupNumber: 
                                        clear_all = False # particle should not be over the robot 
                                        break 
                        else:
                            clear_all = False
                                
                        # if all area is clear, add the coordinate 
                        if clear_all: 
                            grain_outline_ind_list.append(grain_ind)
                        else: 
                            self.color_grain(grain, highlight=False)
                            
                # collect coordinates and highlight outline grains
                xy_list = []
                for grain_ind in grain_outline_ind_list: 
                    grain = self.grain_list[grain_ind]
                    self.color_grain(grain, highlight=highlight)
                    pos = grain.body.position
                    xy_list.append((pos.x, pos.y)) 
                xy_list.sort(key=lambda ee: ee[0]) 
                x_outline = np.array([ee[0] for ee in xy_list])
                y_outline = np.array([ee[1] for ee in xy_list]) + grain.radius
                    
                # remove grains outside the container 
                self.remove_grain_outside_container(space)
            
            # plot outline
            if plot: 
                fig, ax = plt.subplots()
                fig.dpi = 200
                ax.plot(x_outline, y_outline, '*')
                ax.plot(x_outline, np.zeros(x_outline.shape))
                # ax.axis('equal')
                # ax.set_ylim([-10, self.container_height + 10])
                # ax.set_xlim([-10, self.container_width + 10])
                plt.show()
                    
        return x_outline, y_outline
    
    
    def report_pose_all(self): 
        grain_pose_info_list = []
        for grain in self.grain_list: 
            grain_pose_info_list.append(grain.report_pose()) 
            
        return grain_pose_info_list
                
                
    def report_velocity_all(self): 
        velocity_list = []
        for grain in self.grain_list: 
            if grain.body.body_type != pymunk.Body.STATIC: # only report non-zero velocity 
                velocity_list.append(grain.body.velocity.length)
                
        return velocity_list 
                
                
    def plot_grain_particles(self): 
        if len(self.grain_list) > 0: 
            xg = []
            yg = []
            for grain in self.grain_list: 
                xg.append(grain.body.position.x)
                yg.append(grain.body.position.y) 
            fig, ax = plt.subplots()
            fig.dpi = 200 
            ax.plot(xg, yg, '.') 
            
            return fig, ax 

                
        
        
        
        
