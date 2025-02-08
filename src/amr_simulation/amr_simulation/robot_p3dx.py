from amr_simulation.robot import Robot
from typing import Any


class RobotP3DX(Robot):
    """Class to control the Pioneer 3-DX robot."""

    # Constants
    SENSOR_RANGE = 1.0  # Ultrasonic sensor range [m]
    TRACK = 0.33  # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.0975  # Radius of the wheels [m]

    # Sensor location and orientation (x, y, theta) [m, m, rad] in the robot coordinate frame
    SENSORS = [
        (0.1067, 0.1382, 1.5708),
        (0.1557, 0.1250, 0.8727),
        (0.1909, 0.0831, 0.5236),
        (0.2095, 0.0273, 0.1745),
        (0.2095, -0.0273, -0.1745),
        (0.1909, -0.0785, -0.5236),
        (0.1558, -0.1203, -0.8727),
        (0.1067, -0.1382, -1.5708),
        (-0.1100, -0.1382, -1.5708),
        (-0.1593, -0.1203, -2.2689),
        (-0.1943, -0.0785, -2.6180),
        (-0.2129, -0.0273, -2.9671),
        (-0.2129, 0.0273, 2.9671),
        (-0.1943, 0.0785, 2.6180),
        (-0.1593, 0.1203, 2.2689),
        (-0.1100, 0.1382, 1.5708),
    ]

    def __init__(self, sim: Any, dt: float):
        """Pioneer 3-DX robot class initializer.

        Args:
            client_id: CoppeliaSim connection handle.
            dt: Sampling period [s].

        """
        Robot.__init__(self, sim=sim, track=self.TRACK, wheel_radius=self.WHEEL_RADIUS)
        self._dt: float = dt
        self._motors: dict[str, int] = self._init_motors()
        self._sensors: list[int] = self._init_sensors()

    def move(self, v: float, w: float) -> None:
        """Solve inverse differential kinematics and send commands to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        # TODO: 1.1. Complete the function body with your code (i.e., replace the pass statement).
        # Solve inverse differential kinematics
        omega_l = (v - (w * self.TRACK / 2)) / self.WHEEL_RADIUS # left wheel angular velocity
        omega_r = (v + (w * self.TRACK / 2)) / self.WHEEL_RADIUS # right wheel angular velocity

        # Send commands to the motors
        self._sim.setJointTargetVelocity(self._motors["left"], omega_l)
        self._sim.setJointTargetVelocity(self._motors["right"], omega_r)
        
    def sense(self) -> tuple[list[float], float, float]:
        """Read ultrasonic sensors and encoders.

        Returns:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read ultrasonic sensors
        z_us = [float("inf")] * len(self.SENSORS)

        for i in range(len(z_us)):
            is_valid, distance, _, _, _ = self._sim.readProximitySensor(self._sensors[i])

            if is_valid == 1:
                z_us[i] = distance

        # Read encoders
        z_v, z_w = self._sense_encoders()

        return z_us, z_v, z_w

    def _init_motors(self) -> dict[str, int]:
        """Acquire motor handles.

        Returns: {'left': handle, 'right': handle}

        """
        motors: dict[str, int] = {}

        motors["left"] = self._sim.getObject("/Pioneer_p3dx_leftMotor")
        motors["right"] = self._sim.getObject("/Pioneer_p3dx_rightMotor")

        return motors

    def _init_sensors(self) -> list[int]:
        """Acquire ultrasonic sensor handles and initialize US and encoder streaming.

        Returns: List with ultrasonic sensor handles.

        """
        return [
            self._sim.getObject(f"/Pioneer_p3dx_ultrasonicSensor{i + 1}")
            for i in range(len(self.SENSORS))
        ]

    def _sense_encoders(self) -> tuple[float, float]:
        """Solve forward differential kinematics from encoder readings.

        Returns:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read the angular position increment in the last sampling period [rad]
        encoders: dict[str, float] = {}

        encoders["left"] = self._sim.getFloatSignal("leftEncoder")
        encoders["right"] = self._sim.getFloatSignal("rightEncoder")

        # TODO: 1.2. Compute the derivatives of the angular positions to obtain velocities [rad/s].
        omega_l = encoders["left"] / self._dt 
        omega_r = encoders["right"] / self._dt

        # TODO: 1.3. Solve forward differential kinematics (i.e., calculate z_v and z_w).
        z_v = self.WHEEL_RADIUS * (omega_r + omega_l) / 2
        z_w = self.WHEEL_RADIUS * (omega_r - omega_l) / self.TRACK
        
        return z_v, z_w
