import rclpy
import numpy as np

from asyncio            import Future
from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from final_project.ball_funcs      import Ball
from final_project.racket_funcs   import Racket

from final_project.TransformHelpers     import *
import random



#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds
#     1) an update() method to be called regularly by an internal timer,
#     2) a spin() method, aware when a trajectory ends,
#     3) a shutdown() method to stop the timer.
#
#   Take the node name, the update frequency, and the trajectory class
#   as arguments.
#
class GeneratorNode(Node):
    # Initialization.
    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now()+rclpy.time.Duration(seconds=self.dt)

        # Set up a trajectory.
        # self.trajectory = Trajectory(self)
        self.ball = None
        self.racket = Racket(self)
        self.jointnames = self.racket.jointnames()
        
        self.max_side = 2.0
        self.goal = None

        # Add a publisher to send the joint commands.
        self.pub_urdf = self.create_publisher(JointState, '/joint_states', 10)
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub_mark = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)
        self.mark = MarkerArray()
        
        diam = 0.1
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id  = "world"
        self.goal_marker.header.stamp     = self.get_clock().now().to_msg()
        self.goal_marker.action           = Marker.ADD
        self.goal_marker.ns               = "point"
        self.goal_marker.id               = 1
        self.goal_marker.type             = Marker.SPHERE
        self.goal_marker.pose.orientation = Quaternion()
        
        self.goal_marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.goal_marker.color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Create a timer to keep calculating/sending commands.
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))
        
        self.set_goal()
        self.launch_ball()

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.ball.shutdown()
        self.timer.destroy()
        self.destroy_node()

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")
            
    def launch_ball(self):
        # create ball and launch
        self.ball = Ball('balldemo', self.start)
        self.mark.markers.append(self.ball.marker)
        self.racket.set_racket_target(self.ball)
        
    def set_goal(self):
        self.goal = np.array([random.uniform(-self.max_side, self.max_side),
                              random.uniform(-self.max_side, self.max_side),
                              random.uniform(0, self.max_side)]).reshape((3,1))
        
        self.goal_marker.pose.position    = Point_from_p(self.goal)
        self.mark.markers.append(self.goal_marker)
        
        # TODO when ball hits marker get rid of marker (remove from list)
        
        self.racket.set_goal(self.goal)
        print(self.goal)
        
    def check_goal(self):
        # TODO need to check
        ball_p = self.ball.get_position()
        e = 0.01
        if np.linalg.norm(self.goal - ball_p) < e:
            self.mark.markers.remove(self.goal_marker)
            self.ball.shutdown()
            self.ball = None
            return True

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + rclpy.time.Duration(seconds=self.t)

        rac_p = self.racket.get_position()
        rac_orientation_matrix = self.racket.get_orientation()

        if self.ball != None:
            self.ball.update(self.t, self.dt, rac_p, rac_orientation_matrix)
            self.check_goal()
        else:
            self.launch_ball()
        # Compute the desired joint positions and velocities for this time.
        desired = self.racket.evaluate(self.t, self.dt)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired

        # Check the results.
        if not (isinstance(q, list) and isinstance(qdot, list)):
            self.get_logger().warn("(q) and (qdot) must be python lists!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(q) must be same length as jointnames!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(qdot) must be same length as (q)!")
            return
        if not (isinstance(q[0], float) and isinstance(qdot[0], float)):
            self.get_logger().warn("Flatten NumPy arrays before making lists!")
            return


        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg()      # Current time for ROS
        cmdmsg.name         = self.jointnames   # List of joint names
        cmdmsg.position     = q                 # List of joint positions
        cmdmsg.velocity     = qdot              # List of joint velocities
        self.pub_urdf.publish(cmdmsg)
        
        self.pub_mark.publish(self.mark)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



