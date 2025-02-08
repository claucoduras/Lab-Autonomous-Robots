import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

import message_filters
from amr_msgs.msg import PoseStamped, RangeScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

import traceback

from amr_control.wall_follower import WallFollower


class WallFollowerNode(LifecycleNode):
    def __init__(self):
        """Wall follower node initializer."""
        super().__init__("wall_follower")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_localization", False)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles a configuring transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'inactive' state.")

        try:
            # Parameters
            dt = self.get_parameter("dt").get_parameter_value().double_value
            self._enable_localization = (
                self.get_parameter("enable_localization").get_parameter_value().bool_value
            )

            # Subscribers
            # TODO: 1.7. Subscribe to /odom and /us_scan and sync them with _compute_commands_callback.
            self._subscribers: list[message_filters.Subscriber] = []

            # Append as many topics as needed
            self._subscribers.append(
                message_filters.Subscriber(self, Odometry, "/odom")
            )
            self._subscribers.append(
                message_filters.Subscriber(self, RangeScan, "/us_scan")
            )

            # Synchronizer
            ts = message_filters.ApproximateTimeSynchronizer(self._subscribers, 10, 9)
            ts.registerCallback(self._compute_commands_callback)

            # TODO: 4.5. Add /pose to the synced subscriptions only if localization is enabled.
            
            # Publishers
            # TODO: 1.10. Create the /cmd_vel velocity commands publisher (TwistStamped message).
            self._cmd_vel_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)

            # Attribute and object initializations
            self._wall_follower = WallFollower(dt)

        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)

    def _compute_commands_callback(
        self, odom_msg: Odometry, us_msg: RangeScan, pose_msg: PoseStamped = PoseStamped()
    ):
        """Subscriber callback. Executes a wall-following controller and publishes v and w commands.

        Ceases to operate once the robot is localized.

        Args:
            odom_msg: Message containing odometry measurements.
            us_msg: Message containing US sensor readings.
            pose_msg: Message containing the estimated robot pose.

        """
        if not pose_msg.localized:
            # TODO: 1.8. Parse the odometry from the Odometry message (i.e., read z_v and z_w).
            z_v: float = odom_msg.twist.twist.linear.x  # Linear velocity [m/s]
            z_w: float = odom_msg.twist.twist.angular.z  # Angular velocity [rad/s]
            
            # TODO: 1.9. Parse US measurements from the RangeScan message (i.e., read z_us).
            z_us: list[float] = us_msg.ranges
            
            # Execute wall follower
            v, w = self._wall_follower.compute_commands(z_us, z_v, z_w)
            self.get_logger().info(f"Commands: v = {v:.3f} m/s, w = {w:+.3f} rad/s")

            # Publish
            self._publish_velocity_commands(v, w)

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        # TODO: 1.11. Complete the function body with your code (i.e., replace the pass statement).
        # Create the TwistStamped message
        cmd_vel_msg = TwistStamped()

        # Set the linear and angular velocities
        cmd_vel_msg.twist.linear.x = float(v)  # Linear velocity [m/s]
        cmd_vel_msg.twist.angular.z = float(w)  # Angular velocity [rad/s]

        # Publish the message to the /cmd_vel topic
        self._cmd_vel_publisher.publish(cmd_vel_msg)
        
def main(args=None):
    rclpy.init(args=args)
    wall_follower_node = WallFollowerNode()

    try:
        rclpy.spin(wall_follower_node)
    except KeyboardInterrupt:
        pass

    wall_follower_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
