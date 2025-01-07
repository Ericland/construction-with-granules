# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:42:39 2023

@author: ericl
"""
from pymunk import Vec2d

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import time
import pandas as pd

import Utility 


# In[]
class Robot_Kinematics: 
    """
    Compute SAW robot kinematics
    """
    def __init__(
            self, 
            link_num=11, 
            link_length=14, 
            link1_length=28, 
            helix_L=120, 
            helix_A=22.5, 
            freq=1, 
            dt=0.1, 
            phi=0, 
            right_tail=True, 
            advancing_wave=True, 
                 ): 
        # robot dimension
        self.link_num = link_num
        self.link_length = link_length
        self.link1_length = link1_length
        self.helix_L = helix_L
        self.helix_A = helix_A
        
        # time variables
        self.freq = freq
        self.dt = dt 
        self.phi = phi
        
        # motion parameters
        self.right_tail = right_tail
        self.advancing_wave = advancing_wave
        
        # kinematics solution parameters  
        self.sol = {}
        self.update_coef()
        
        # Check input
        assert freq > 0, "frequency should be positive"
        
        
    def update_coef(self): 
        """
        compute period, time step and list of sampling time

        Returns
        -------
        None.

        """
        # sampling parameters 
        self.T = 1 / self.freq
        self.sample_num = int(self.T / self.dt)
        self.time_list = self.dt * np.arange(self.sample_num + 1)
        
        # sine wave parameters
        self.k = 2 * np.pi / self.helix_L
        if self.advancing_wave: 
            self.w = 2 * np.pi * self.freq
        else: 
            self.w = -2 * np.pi * self.freq
        
    
    def f(self, x, t): 
        """
        f(x, t) = A * sin(k*x - w*t + phi)
    
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.
        phi : TYPE, optional
            DESCRIPTION. The default is 0.
        freq : TYPE, optional
            DESCRIPTION. The default is 1.
    
        Returns
        -------
        None.
    
        """
        return self.helix_A * np.sin(self.k * x - self.w * t + self.phi)
    
    
    def df(self, x, t): 
        """
        f(x, t) = A * sin(k*x - w*t + phi)
    
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.
        phi : TYPE, optional
            DESCRIPTION. The default is 0.
        freq : TYPE, optional
            DESCRIPTION. The default is 1.
    
        Returns
        -------
        None.
    
        """
        return self.helix_A * self.k * np.cos(self.k * x - self.w * t + self.phi)
    
    
    def solve_link1(self, f, df): 
        """
        Solve for the 1st link that is fixed to the motor house at (0, 0)
    
        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        if self.right_tail: 
            minStep = np.cos(np.arcsin(self.helix_A / self.link1_length)) * self.link1_length
        else:
            minStep = -np.cos(np.arcsin(self.helix_A / self.link1_length)) * self.link1_length
        
        def f_root(x): 
            return x**2 + f(x)**2 - self.link1_length**2
        
        def jac(x): 
            return 2 * x + 2 * f(x) * df(x)
        
        sol = root(f_root, minStep, jac=jac)
        if not sol.success: 
            print(sol.x, f_root(sol.x), sol.success)
            raise Exception("Fail to solve for 1st link!")
            
        return (sol.x[0], f(sol.x[0]))
    
    
    def solve_links(self, f, df, x0): 
        """
        Solve for the end points of links
    
        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        x0 : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        if self.right_tail: 
            minStep = np.cos(np.arcsin(self.helix_A / self.link1_length)) * self.link1_length
        else:
            minStep = -np.cos(np.arcsin(self.helix_A / self.link1_length)) * self.link1_length
        
        links = []
        x_cur = x0
        links.append((x_cur, f(x_cur)))
        
        def f_root(x): 
            return (x - x_cur)**2 + (f(x) - f(x_cur))**2 - self.link_length**2
        
        def jac(x): 
            return 2 * (x - x_cur) + 2 * (f(x) - f(x_cur)) * df(x)
        
        for ii in range(self.link_num):         
            sol = root(f_root, x_cur + minStep, jac=jac)
            if not sol.success: 
                print(sol.x, x_cur, f_root(sol.x), sol.success)
                raise Exception("Fail to solve for links")
            x_cur = sol.x[0]
            links.append((x_cur, f(x_cur)))
            
        return links
    
    
    def link2vec(self, links): 
        """
        Convert list of link end points to list of Vec2d vectors
        """
        vecs = []
        for ii in range(len(links) - 1): 
            x1, y1 = links[ii]
            x2, y2 = links[ii+1]
            vec = Vec2d(x2-x1, y2-y1)
            vecs.append(vec)
        
        return vecs
    
    
    def compute_abs_link_angles(self, vecs): 
        angles = []
        for vec in vecs: 
            angle = vec.angle
            angles.append(angle)
            
        return angles
    
    
    def compute_relative_link_angles(self, vecs): 
        angles = [vecs[0].angle]
        for ii in range(len(vecs) - 1): 
            vec1 = vecs[ii]
            vec2 = vecs[ii+1]
            angle = vec1.get_angle_between(vec2)
            angles.append(angle)
            
        return angles
    
    
    def compute_change_of_relative_angles(self, theta1, theta2): 
        """
        theta1 is the current angle and theta2 is the future projected angle
        input must be numpy arrays 
        """
        diff = theta2 - theta1
        dtheta = Utility.rad2pipi(diff)
        
        return dtheta
    
    
    def plot_links(self, pause=False): 
        """
        Plot links over time

        Returns
        -------
        None.

        """
        if self.sol != {}: 
            for t in self.time_list: 
                fig, ax = plt.subplots()
                ax.axis('equal')
                if self.right_tail: 
                    ax.set_xlim([-self.link_length, self.helix_L * 1.5 + self.link_length])
                else: 
                    ax.set_xlim([-self.helix_L * 1.5 - self.link_length, self.link_length])
                ax.set_ylim([-self.helix_A * 1.1, self.helix_A * 1.1])
                
                links_t = self.sol['xy'][str(t)]
                x_list = [ee[0] for ee in links_t]
                y_list = [ee[1] for ee in links_t]
                ax.plot(x_list, y_list, "*--")
                ax.set(title=str(t))
                
                plt.show()
                if pause: time.sleep(0.1)
            
            
    def compute(self): 
        """
        Compute:
            link end points (xy)
            absolute link angles (Theta)
            relative link angles (theta)
            change of relative link angles (dtheta)

        Returns
        -------
        None.

        """
        links_dict = {}
        Theta_dict = {}
        theta_dict = {}
        for t in self.time_list: 
            def ft(x): 
                return self.f(x, t)
            
            def dft(x): 
                return self.df(x, t)
            
            link1 = self.solve_link1(ft, dft)
            links_t = self.solve_links(ft, dft, x0=link1[0])
            links_t.insert(0, (0.0, 0.0))
            links_dict[str(t)] = links_t
            vecs_t = self.link2vec(links_t)
            Theta_t = self.compute_abs_link_angles(vecs_t)
            theta_t = self.compute_relative_link_angles(vecs_t)
            Theta_dict[str(t)] = Theta_t
            theta_dict[str(t)] = theta_t 
            
        df0 = pd.DataFrame(data=links_dict) # xy
        df1 = pd.DataFrame(data=Theta_dict) # absolute link angles Theta
        df2 = pd.DataFrame(data=theta_dict) # relative link angles theta
        
        theta_array = df2.to_numpy()
        m, n = theta_array.shape
        dtheta_dict = {}
        for ii in range(n-1): 
            theta1 = theta_array[:,ii]
            theta2 = theta_array[:,ii+1]
            dtheta_dict[str(self.time_list[ii+1])] = self.compute_change_of_relative_angles(theta1, theta2)
        df3 = pd.DataFrame(data=dtheta_dict) # change of relative link angle dtheta
        
        sol = {
            "xy": df0, 
            "Theta": df1, 
            "theta": df2, 
            "dtheta": df3
            }
        self.sol = sol
                
        return sol
    
    
    def compute_kinematics(self): 
        """
        Compute kinematics over one period

        Returns
        -------
        None.

        """
        self.update_coef() # in case of change of coef
        sol = self.compute()
        
        return sol
    
    
    def compute_kinematics_over_dt(self, dphi=0): 
        """
        Compute kinematics for just one dt at phase = initial phase + dphi
        Save computation time when kinematics in short time is needed. 

        Parameters
        ----------
        phi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.update_coef() # in case of change of coef 
        self.time_list = self.dt * np.arange(2)
        phase_init = self.phi
        self.phi += dphi
        sol = self.compute()
        
        # set everything back 
        self.update_coef()
        self.phi = phase_init
        
        return sol



# In[]
if __name__ == "__main__": 
    run = 1
    if run: 
        # right tail advancing 
        sawk = Robot_Kinematics(link_num=11, dt=0.1)
        sol1 = sawk.compute_kinematics()
        sawk.plot_links()
    
    run = 0
    if run: 
        # right tail advancing 
        sawk = Robot_Kinematics()
        sol1 = sawk.compute_kinematics()
        sawk.plot_links()
        
        # right tail reversing
        sawk.advancing_wave = False
        sol2 = sawk.compute_kinematics()
        sawk.plot_links()
        
        # left tail reversing
        sawk.right_tail = False
        sol3 = sawk.compute_kinematics()
        sawk.plot_links()
        
        # left tail advancing 
        sawk.advancing_wave = True
        sol4 = sawk.compute_kinematics()
        sawk.plot_links()
    
    run = 0
    if run: 
        sawk = Robot_Kinematics()
        sol1 = sawk.compute_kinematics()
        d1 = sol1['xy'].iloc[:, 0]
        d1x = np.array([ee[0] for ee in d1])
        d1y = np.array([ee[1] for ee in d1])
        
        sawk.right_tail = False
        sol2 = sawk.compute_kinematics()
        d2 = sol2['xy'].iloc[:, 0]
        d2x = np.array([ee[0] for ee in d2])
        d2y = np.array([ee[1] for ee in d2])
        
        
    run = 0
    if run: 
        sawk = Robot_Kinematics()
        sol_T = sawk.compute_kinematics()
        w = sawk.w
        dt = sawk.dt
        sample_num = sawk.sample_num
        phi_init = sawk.phi
        dtheta_list = []
        for ii in range(sample_num): 
            dphi = -w * dt * ii
            sol_t = sawk.compute_kinematics_over_dt(dphi)
            dtheta_list.append(sol_t['dtheta'].to_numpy())
        dtheta_joined = np.block(dtheta_list)
        dtheta_T = sol_T['dtheta'].to_numpy()
        print(np.sum(np.abs(dtheta_T - dtheta_joined)))
        print('max rotation angle (deg):', np.amax(np.abs(np.degrees(dtheta_T))))
        
        
    run = 0
    if run: 
        sawk = Robot_Kinematics()
        sol_T = sawk.compute_kinematics()
        sol_t = sawk.compute_kinematics_over_dt()

        



