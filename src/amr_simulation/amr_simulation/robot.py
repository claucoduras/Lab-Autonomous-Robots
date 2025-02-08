from abc import ABC, abstractmethod
from typing import Any


class Robot(ABC):
    """Abstract base class to control mobile robots."""

    def __init__(self, sim: Any, track: float, wheel_radius: float):
        """Robot class initializer.

        Args:
            sim: CoppeliaSim simulation handle.
            track: Distance between the centerline of two wheels on the same axle [m].
            wheel_radius: Radius of the wheels [m].

        """
        self._sim = sim
        self._track = track
        self._wheel_radius = wheel_radius

    @abstractmethod
    def move(self, v: float, w: float):
        """Solve inverse kinematics and send commands to the motors.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        pass

    @abstractmethod
    def sense(self):
        """Acquire sensor readings."""
        pass
