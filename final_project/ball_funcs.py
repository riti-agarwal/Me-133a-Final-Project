import rclpy
import numpy as np

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from final_project.TransformHelpers     import *

class Ball(Node):
    # Initialization.
    def __init__(self, name, start):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.033
        self.side = 2.0 # distance between bounces
        
        self.init_p = np.array([0.0, 0.0, self.radius]).reshape((3,1))
        self.init_v = np.array([1.0, 0.1,  5.0       ]).reshape((3,1))

        self.p = self.init_p
        self.v = self.init_v
        self.a = np.array([0.0, 0.0, -9.81      ]).reshape((3,1))
        
         # racket_collision_distance - how close ball needs to be to tennis racket for a collision to be detected.
        # used as a safety margin so a collision is detected even if the ball and racket are not perfectly aligned.
        self.racket_dist = 0.05

        # Create the sphere marker.
        diam        = 2 * self.radius
        self.marker = Marker()
        self.marker.header.frame_id  = "world"
        self.marker.header.stamp     = self.get_clock().now().to_msg()
        self.marker.action           = Marker.ADD
        self.marker.ns               = "point"
        self.marker.id               = 2
        self.marker.type             = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.pose.position    = Point_from_p(self.p)
        self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.marker.color            = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        # a = 0.8 is slightly transparent!

        # Create the marker array message.
        # self.mark = MarkerArray()
        # self.mark.markers.append(self.marker)

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        # self.dt    = 1.0 / float(rate)
        # self.t     = -self.dt
        self.start = start

        # Create a timer to keep calling update().
        # self.create_timer(self.dt, self.update)
        # self.get_logger().info("Running with dt of %f seconds (%fHz)" %
        #                        (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    def get_random_path():
        # TODO
        return None

    def get_position(self):
        return self.p
    
    def get_direction(self):
        return get_direction_from_v(self.v)
    
    def get_pd_at_y(self, given_y = 0):
        # p = p0 + v0t + 1/2 a t^2
        # given_y = y0 + vy0 t
        t = (given_y - self.init_p[1, 0]) / self.init_v[1, 0]
        v = self.init_v + self.a * t
        d = get_direction_from_v(v)
        p = self.init_p + self.init_v * t + self.a * (t ** 2) / 2
        return p, d, t

    # Update - send a new joint command every time step.
    def update(self, t, dt, rac_p, rac_orientation_matrix):
        # Integrate the velocity, then the position.
        self.v += dt * self.a
        self.p += dt * self.v

        # can change this to0 check for collision
        # Check for a bounce - not the change in x velocity is non-physical.
        if self.p[2,0] < self.radius:
            self.p[2,0] = self.radius + (self.radius - self.p[2,0])
            # changing the velocity in the z direction, so velocity is in the other direction
            self.v[2,0] *= -1.0
            # changing the velocity in the x direction
            # self.v[0,0] *= -1.0   # Change x just for the fun of it!

        # Update the ID number to create a new ball and leave the
        # previous balls where they are.
        #####################
        # self.marker.id += 1
        #####################

        # Check for a bounce on the side wall
        # self.side is the width of the side wall
        # If a collision occurs, the ball's position is adjusted to be just outside the wall, 
        # and the x-axis component of the velocity is reversed to simulate a bounce.
        if abs(self.p[0, 0]) + self.radius > self.side / 2.0:
            # Bounce back from the side wall
            self.p[0, 0] = np.sign(self.p[0, 0]) * (self.side / 2.0 - self.radius)
            self.v[0, 0] *= -1.0
            # print("wall collision")

        # if np.linalg.norm(self.p - rac_p) < self.radius + racket_collision_distance:
        #     # Bounce back from the tennis racket
        #     print("hello world")
        #     direction_to_racket = (rac_p - self.p) / np.linalg.norm(rac_p - self.p)
        #     self.p = rac_p - (self.radius + racket_collision_distance) * direction_to_racket
        #     self.v = -self.v + 2 * np.dot(self.v.T, direction_to_racket) * direction_to_racket

       

        # Check for a collision with the tennis racket
        if np.linalg.norm(self.p - rac_p) < self.radius + self.racket_dist:
            # print ("collision happened")
            # Bounce back from the tennis racket
            # self.p = rac_p + np.dot(rac_orientation_matrix, np.array([[0], [0], [-self.radius - racket_collision_distance]]))
            # self.v = -self.v + 2 * np.dot(self.v.T, direction_to_racket) * direction_to_racket
            relative_position = rac_p - self.p
            collision_normal = np.dot(rac_orientation_matrix.T, relative_position)
            collision_normal = collision_normal / np.linalg.norm(collision_normal)
            reflection_direction = - self.v + 2 * np.dot(self.v.T, collision_normal) * collision_normal
            self.v = reflection_direction
            self.p = rac_p - (self.radius + self.racket_dist) * collision_normal


        # Update the message and publish.
        now = self.start + Duration(seconds=t)
        self.marker.header.stamp  = now.to_msg()
        self.marker.pose.position = Point_from_p(self.p)
        # self.pub.publish(self.mark)


