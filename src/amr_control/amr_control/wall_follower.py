import numpy as np


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    def __init__(self, dt: float):
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt

    def compute_commands(self, z_us: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """

        # TODO: 1.14. Complete the function body with your code (i.e., compute v and w).
        # Key parameters for speed and distances.
        desired_distance = 0.35  # Distance from the wall
        max_speed = 0.6  # Full speed
        turn_speed = 0.65  # Speed for normal turns
        sharp_turn_speed = 1.9  # Speed for tight 180-degree turns
        threshold_front = 1.4  # Obstacle reaction distance
        turn_around_threshold = 0.6  # Threshold for dead-ends

        # PD gains
        Kp_left = 0.9  # Left wall-following
        Kp_right = 1.3  # Reduced from 1.7 to prevent overcorrection
        Kd = 1.2  # Slightly reduced to smooth out oscillations

        alpha = 0.6  # Low-pass filter coefficient for derivative smoothing
        error_deadband = 0.05  # Ignore small oscillations

        # Ultrasonic readings
        front = min(z_us[3], z_us[4])
        left_side = min(z_us[0], z_us[1], z_us[2])
        right_side = min(z_us[5], z_us[6], z_us[7])

        # Default movement
        v = max_speed
        w = 0.0

        # U-turn logic
        if front < 0.4 and (left_side < 0.4 or right_side < 0.4):
            v = 0.03  # Slow forward
            w = sharp_turn_speed if left_side > right_side else -sharp_turn_speed
            return v, w

        # Obstacle avoidance (turn when blocked)
        if front < threshold_front:
            v = 0.25  # Slow down
            if right_side > left_side + 0.1:  # Favor right turns
                w = -turn_speed * 1.3  # Sharper right turn
            else:
                w = turn_speed  # Left turn
            return v, w

        # Wall-following PD controller
        if left_side < right_side:
            error = left_side - desired_distance
            Kp = Kp_left
        else:
            error = desired_distance - right_side
            Kp = Kp_right

        # Apply deadband (ignore tiny oscillations)
        if abs(error) < error_deadband:
            error = 0

        # Smoothed derivative calculation
        last_error = getattr(self, "_last_error", 0)
        last_d_error = getattr(self, "_last_d_error", 0)
        dt = self._dt

        raw_d_error = (error - last_error) / dt
        d_error = alpha * raw_d_error + (1 - alpha) * last_d_error  # Low-pass filter

        self._last_error = error
        self._last_d_error = d_error

        # Compute angular velocity
        w = Kp * error + Kd * d_error

        # Special behavior for open areas
        if front > 0.8 and left_side > 0.8 and right_side > 0.1:
            v = 0.02
            w = sharp_turn_speed if left_side > right_side else -sharp_turn_speed
            return v, w

        return v, w
