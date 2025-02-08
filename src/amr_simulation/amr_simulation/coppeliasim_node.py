import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

import message_filters
from amr_msgs.msg import PoseStamped, RangeScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

import math
import traceback
from transforms3d.euler import quat2euler

from amr_simulation.coppeliasim import CoppeliaSim
from amr_simulation.robot_p3dx import RobotP3DX


class CoppeliaSimNode(LifecycleNode):
    def __init__(self):
        """Simulator node initializer."""
        super().__init__("coppeliasim")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_localization", False)
        self.declare_parameter("goal", (float("inf"), float("inf")))
        self.declare_parameter("goal_tolerance", 0.15)

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
            self._goal = tuple(
                self.get_parameter("goal").get_parameter_value().double_array_value.tolist()
            )
            goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value

            self.declare_parameter("start", (0.0, 0.0, 0.0))
            start = tuple(
                self.get_parameter("start").get_parameter_value().double_array_value.tolist()
            )

            # Subscribers
            # TODO: 1.12. Subscribe to /cmd_vel. Connect it with with _next_step_callback.
            self.create_subscription(
                TwistStamped,
                "/cmd_vel",
                self._next_step_callback,  # Puede ir dentro de la suscripción porque solo es un mensaje. Si tuvieses que esperar a recibir más de uno tendrías que hacerlo fuera de la suscripción.
                10,
            )

            # TODO: 2.3. Synchronize the /pose and /cmd_vel subscribers if enable_localization is True.
            if self._enable_localization:
                # Create synchronized subscribers for /cmd_vel and /pose
                self._subscribers: list[message_filters.Subscriber] = []
                self._subscribers.append(message_filters.Subscriber(self, PoseStamped, "/pose"))
                self._subscribers.append(message_filters.Subscriber(self, TwistStamped, "/cmd_vel"))

                # Synchronize messages using ApproximateTimeSynchronizer
                ts = message_filters.ApproximateTimeSynchronizer(
                    self._subscribers, queue_size=10, slop=0.1
                )
                ts.registerCallback(self._next_step_callback)

            # Publishers
            # TODO: 1.4. Create the /odom (Odometry message) and /us_scan (RangeScan) publishers.
            self._odom_publisher = self.create_publisher(Odometry, "/odom", 10)
            self._us_scan_publisher = self.create_publisher(RangeScan, "/us_scan", 10)

            # Attribute and object initializations
            self._coppeliasim = CoppeliaSim(dt, start, goal_tolerance)
            self._robot = RobotP3DX(self._coppeliasim.sim, dt)
            self._localized = False

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

        # Initial method calls
        self._next_step_callback(cmd_vel_msg=TwistStamped())

        return super().on_activate(state)

    def __del__(self):
        """Destructor."""
        try:
            self._coppeliasim.stop_simulation()
        except AttributeError:
            pass

    def _next_step_callback(self, cmd_vel_msg: TwistStamped, pose_msg: PoseStamped = PoseStamped()):
        """Subscriber callback. Executes a simulation step and publishes the new measurements.

        Args:
            cmd_vel_msg: Message containing linear (v) and angular (w) speed commands.
            pose_msg: Message containing the estimated robot pose.

        """
        # Check estimated pose
        self._check_estimated_pose(pose_msg)

        # TODO: 1.13. Parse the velocities from the TwistStamped message (i.e., read v and w).
        v: float = cmd_vel_msg.twist.linear.x
        w: float = cmd_vel_msg.twist.angular.z

        # Execute simulation step
        self._robot.move(v, w)
        self._coppeliasim.next_step()
        z_us, z_v, z_w = self._robot.sense()

        # Check goal
        if self._check_goal():
            return

        # Publish
        self._publish_odometry(z_v, z_w)
        self._publish_us(z_us)

    def _check_estimated_pose(self, pose_msg: PoseStamped = PoseStamped()) -> None:
        """If the robot is localized, compares the estimated and real poses.

        Outputs a ROS log message to the Terminal with the estimated pose upon localization and
        another with the real and estimated values thereafter for monitoring purposes.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """
        self._localized = pose_msg.localized

        if self._localized:
            x_h = pose_msg.pose.position.x
            y_h = pose_msg.pose.position.y
            quat_w = pose_msg.pose.orientation.w
            quat_x = pose_msg.pose.orientation.x
            quat_y = pose_msg.pose.orientation.y
            quat_z = pose_msg.pose.orientation.z

            _, _, th_h = quat2euler((quat_w, quat_x, quat_y, quat_z))
            th_h %= 2 * math.pi
            th_h_deg = math.degrees(th_h)

            real_pose, position_error, within_tolerance = self._coppeliasim.check_position(x_h, y_h)
            x, y, th = real_pose
            th %= 2 * math.pi
            th_deg = math.degrees(th)

            self.get_logger().warn(
                f"Localized at x = {x_h:.2f} m, y = {y_h:.2f} m, "
                f"theta = {th_h:.2f} rad ({th_h_deg:.1f}º) | "
                f"Real pose: x = {x:.2f} m, y = {y:.2f} m, theta = {th:.2f} rad ({th_deg:.1f}º) | "
                f"Error = {position_error:.3f} m {'(OK)' if within_tolerance else ''}",
                once=True,  # Log only the first time this function is hit
            )

            self.get_logger().info(
                f"Estimated: x = {x_h:.2f} m, y = {y_h:.2f} m, "
                f"theta = {th_h:.2f} rad ({th_h_deg:.1f}º) | "
                f"Real pose: x = {x:.2f} m, y = {y:.2f} m, theta = {th:.2f} rad ({th_deg:.1f}º) | "
                f"Error = {position_error:.3f} m {'(OK)' if within_tolerance else ''}",
                skip_first=True,  # Log all but the first time this function is hit
            )

    def _check_goal(self) -> bool:
        """Checks whether the robot is localized and has reached the goal within tolerance or not.

        Returns:
            bool: True if the condition is met; False otherwise.

        """
        goal_found = False

        if self._localized:
            _, _, goal_found = self._coppeliasim.check_position(self._goal[0], self._goal[1])

            if goal_found:
                self.get_logger().warn("Congratulations, you reached the goal!")
                execution_time, simulated_time, steps = self._coppeliasim.stop_simulation()
                self._print_statistics(execution_time, simulated_time, steps)

        return goal_found

    def _print_statistics(self, execution_time: float, simulated_time: float, steps: int) -> None:
        """Outputs a ROS log message to the Terminal with a summary of timing statistics.

        Args:
            execution_time: Natural (real) time taken to localize and reach the goal.
            simulated_time: Simulation time taken to finish the challenge.
            steps: Number of steps (simulated_time / dt).

        """
        try:
            self.get_logger().warn(
                f"Execution time: {execution_time:.3f} s ({execution_time / steps:.3f} s/step) | "
                f"Simulated time: {simulated_time:.3f} s ({steps} steps)"
            )
        except ZeroDivisionError:
            pass

    def _publish_odometry(self, z_v: float, z_w: float) -> None:
        """Publishes odometry measurements in a nav_msgs.msg.Odometry message.

        Args:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # TODO: 1.5. Complete the function body with your code (i.e., replace the pass statement).
        # Create the odometry message
        odom_msg = Odometry()

        # Set velocity data (linear and angular)
        odom_msg.twist.twist.linear.x = z_v
        odom_msg.twist.twist.angular.z = z_w

        # Publish the message to the /odom topic
        self._odom_publisher.publish(odom_msg)

    def _publish_us(self, z_us: list[float]) -> None:
        """Publishes US measurements in a custom amr_msgs.msg.RangeScan message.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].

        """
        # TODO: 1.6. Complete the function body with your code (i.e., replace the pass statement).
        # Create the RangeScan message
        us_scan_msg = RangeScan()

        # Set the range data from the ultrasonic sensors
        us_scan_msg.ranges = z_us  # List of distances from each sensor to the nearest obstacle

        # Publish the message to the /us_scan topic
        self._us_scan_publisher.publish(us_scan_msg)


def main(args=None):
    rclpy.init(args=args)
    coppeliasim_node = CoppeliaSimNode()

    try:
        rclpy.spin(coppeliasim_node)
    except KeyboardInterrupt:
        pass

    coppeliasim_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
