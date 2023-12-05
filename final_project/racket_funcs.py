from enum import Enum
import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from final_project.GeneratorNode      import GeneratorNode
from final_project.TransformHelpers   import *
from final_project.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from final_project.KinematicChain     import KinematicChain

class state(Enum):
    TOTARGET = 1
    WAITINGTARGET = 2
    TOINIT = 3
    WAITINGINIT = 4

class Racket():
    def __init__(self, node, goal = None):
        self.goal = goal
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())
        
        # Define the various points.
        self.q0 = np.radians(np.array([0, 90, -90, 0, 0, 0]).reshape((-1,1)))
        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((-1,1))
        self.R0 = Reye() @ Rotx(-pi/2) @ Roty(-pi/2)
        
        # TODO from ball trajectory, get position and normal orientation
        self.p_target = np.array([0.5, 0.5, 0.15]).reshape((-1,1))
        self.r_target = Reye() @ Rotx(-pi/2) @ Roty(-pi/2)
        
        self.target_changed = False
        
        self.duration = 2.5
        self.last_time = 0
        self.state = state.WAITINGINIT

        self.rac_radius = 0.2
        self.lamb = 20
        self.q  = self.q0
        self.p = self.p0
        self.R = self.R0

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
    
    def set_racket_target(self, ball):
        ball_p, ball_d, t = ball.get_pd_at_y(given_y = 0)
        self.p_target = ball_p
        if self.goal.all() == None:
            # self.r_target = ball_d
            self.r_target = self.R0
        else:
            to_goal = self.goal - ball_p
            r_vec = cross(to_goal, ball_d)
            # self.r_target = R_from_quat(quat_from_euler(r_vec))
            self.r_target = self.R0
        # TODO within ball trajectory
        self.target_changed = True
        # self.duration = t
        print(self.p_target, self.r_target, self.duration)
    
    def checkwaiting(self, t):
        if self.state == state.WAITINGINIT and self.target_changed:
            self.state = state.TOTARGET
            self.last_time = t
            self.target_changed = False
        elif self.state == state.WAITINGTARGET:
            # if ball hit
            self.state = state.TOINIT
            self.last_time = t
            
    def get_position(self):
        return self.p

    def get_orientation(self):
        return self.R
    
    def set_goal(self, goal):
        self.goal = goal

    def get_radius(self):
        return self.rac_radius

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if self.state == state.WAITINGINIT or self.state == state.WAITINGTARGET:
            q = self.q
            qdot = np.zeros((6, 1))
            self.checkwaiting(t)
        else:
            if self.state == state.TOTARGET:
                if t - self.last_time >= self.duration:
                    self.last_time = t
                    q = self.q
                    qdot = np.zeros((6, 1))
                    self.state = state.WAITINGTARGET
                    return (q.flatten().tolist(), qdot.flatten().tolist())
                p0 = self.p0
                pf = self.p_target
                r0 = self.R0
                rf = self.r_target
            elif self.state == state.TOINIT:
                if t - self.last_time >= self.duration:
                    self.last_time = t
                    q = self.q
                    qdot = np.zeros((6, 1))
                    self.state = state.WAITINGINIT
                    return (q.flatten().tolist(), qdot.flatten().tolist())
                p0 = self.p_target
                pf = self.p0
                r0 = self.r_target
                rf = self.R0
                
            t = fmod(t - self.last_time, self.duration)  
            e = ex() + ey() + ez()
            alpha = pi / 2
            
            (s0, s0dot) = goto(t, 2.5, 0.0, 1.0)

            pd = p0 + (pf - p0) * s0
            vd =      (p0 - pf) * s0dot

            Rd = r0 @ (s0 * np.linalg.inv(r0)) @ rf
            alphadot = alpha * s0dot
            wd = alphadot * e

            qlast = self.q
            (p, R, Jv, Jw) = self.chain.fkin(qlast)
            J = np.vstack((Jv, Jw))
            V = np.vstack((vd, wd))
            E = np.vstack((ep(pd, p), eR(Rd, R)))
            
            qdot = np.linalg.pinv(J) @ (V + self.lamb * E)
            
            q = qlast + dt * qdot
            
            # Update
            self.q = q
            self.p = p
            self.R = R
            
            # TODO Add secondary tasks 

        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())