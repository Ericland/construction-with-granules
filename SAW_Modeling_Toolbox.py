# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:32:14 2023

@author: ericl
"""


# In[]
import pygame
import pymunk
import pymunk.autogeometry
import pymunk.pygame_util
# from pymunk import Vec2d

import random
random.seed(5)  # try keep difference the random factor the same each run.
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
from PIL import Image
# from tqdm import tqdm
# import os
# import cv2

import Utility


# In[]
class Visualization_by_Pygame: 
    def __init__(self, 
                 screen_size=(1000, 1000), 
                 fps=100, 
                 ): 
        # initialize screen
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen) # Let's quickly draw whole space together 
        pymunk.pygame_util.positive_y_is_up = True # positive y pointing down 
        
        # set timing
        self.clock = pygame.time.Clock() # initialize a timer for updating screen
        self.fps = fps
        
        # gif animation parameters
        self.gif_on = False
        self.gif_frame_num = 100 # number of frames in the gif
        self.gif_counter = 0
        self.images = []
        
        
    def stop(self): 
        """
        Stop simulation

        Returns
        -------
        None.

        """
        pygame.quit()
        msg = "stop"
        
        return msg
        
        
    def check_event(self): 
        """
        Check user input

        Returns
        -------
        None.

        """
        msg = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                msg = self.stop()
            elif event.type == pygame.KEYDOWN: 
                if event.key in [pygame.K_ESCAPE, pygame.K_q]: 
                    msg = self.stop()
                elif event.key == pygame.K_p:
                    save_loc, file_name = self.save_screenshot()
                    msg = "image"
                    print("A screenshot is saved: " + save_loc + file_name)
                elif event.key == pygame.K_g: 
                    self.gif_on = True 
                    msg = "GIF"
                    print("Start generating GIF!")
                elif event.key == pygame.K_h: 
                    if self.gif_on and self.gif_counter < self.gif_frame_num-1: 
                        self.gif_counter = self.gif_frame_num-1
                        msg = "stop GIF"
                    
        return msg
                
                
    def update(self, space, info_dict, draw=True): 
        """
        Visualize a given pymunk space

        Parameters
        ----------
        space : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if draw: 
            # Draw everything on screen
            # self.draw_options.transform = pymunk.Transform.scaling(0.5) # scale everythong if necessary
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES # draw shapes
            self.draw_options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS # draw collisions
            self.screen.fill(pygame.Color("white")) # background color
            space.debug_draw(self.draw_options) # Draw the current state of the space on screen
            
            # print content from info_dict
            font = pygame.font.Font(None, 16)
            for ii, item in enumerate(info_dict.items()): 
                text = item[0] + ": " + str(item[1])
                self.screen.blit(font.render(text, 1, pygame.Color("black")), (5, 5 + ii * 20))
            inst_dict = [
                "press 'P' to save image", 
                "press 'G' to generate GIF", 
                "press 'H' to stop GIF", 
                ]
            idl = len(info_dict)
            for jj in range(len(inst_dict)): 
                self.screen.blit(font.render(inst_dict[jj], 1, pygame.Color("black")), (5, 5 + (idl+jj+1) * 20))
                
            # Update the full display Surface to the screen 
            pygame.display.flip() 
            
            # Make gif
            self.make_gif()
        
            # Update total runing time
            self.clock.tick(self.fps) 
            
        else:
            # Hide window
            pygame.display.iconify()
        
        
    def make_gif(self): 
        """
        Make a short GIF animation 

        Returns
        -------
        None.

        """
        if self.gif_on: 
            # store images 
            strFormat = 'RGBA'
            raw_str = pygame.image.tostring(self.screen, strFormat, False)
            image = Image.frombytes(strFormat, self.screen.get_size(), raw_str)
            self.images.append(image)
            self.gif_counter += 1
            
            # once number of stored images reaches the desired limit, generate gif
            if self.gif_counter == self.gif_frame_num: 
                frame_num = len(self.images)
                self.images[0].save(
                    "img\saw.gif",
                    save_all=True, 
                    append_images=self.images[1:],
                    optimize=True, 
                    duration=frame_num*10//self.fps, 
                    loop=0, 
                    )
                print("A GIF is saved!")
                # reset gif parameters 
                self.gif_on = False
                self.gif_counter = 0
                self.images = []
                
                
    def save_screenshot(self, save_loc='img/', file_name='default'):
        if file_name == 'default': 
            file_time = Utility.get_time_str()
            file_name = 'img_' + file_time + '.png'
        file_dir = save_loc + file_name
        pygame.image.save(self.screen, file_dir)
        
        return save_loc, file_name
                
                
# In[]


