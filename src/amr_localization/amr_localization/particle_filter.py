import datetime
import math
import numpy as np
import os
import pytz

from amr_localization.map import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensors: list[tuple[float, float, float]],
        sensor_range: float,
        particle_count: int,
        sigma_v: float = 0.15,
        sigma_w: float = 0.75,
        sigma_z: float = 0.25,
        logger=None,
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sensors: Robot sensors' pose in the robot coordinate frame (x, y, theta) [m, m, rad].
            sensor_range: Sensor measurement range [m].
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensors: list[tuple[float, float, float]] = sensors
        self._sensor_range: float = sensor_range
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0
        self._logger = logger

        self._map = Map(map_path, sensor_range, compiled_intersect=True, use_regions=True)
        self._particles = self._init_particles(particle_count)
        self._ds, self._phi = self._init_sensor_polar_coordinates(sensors)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # # TODO: 2.10. Complete the missing function body with your code.
        # # ya es numpy array
        # particles = self._particles
        
        # # Apply DBSCAN clustering
        # db = DBSCAN(eps=0.1, min_samples=5).fit(particles[:, :2])  # Only cluster on x, y coordinates
        # labels = db.labels_
        
        # # Count the number of clusters (ignoring noise if any)
        # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # # If there is only one cluster, estimate the pose from the cluster centroid
        # if n_clusters == 1:
        #     # Get the indices of the particles in the largest cluster
        #     cluster_indices = np.where(labels == 0)[0]
        #     cluster_particles = particles[cluster_indices]
            
        #     # Compute the centroid of the cluster
        #     centroid = np.mean(cluster_particles[:, :2], axis=0)
            
        #     # Compute the average orientation (theta) of the cluster
        #     # Handle the wrap-around at 0 and 2π
        #     theta = np.arctan2(np.mean(np.sin(cluster_particles[:, 2])), np.mean(np.cos(cluster_particles[:, 2])))
            
        #     # Set the pose estimate
        #     pose = (centroid[0], centroid[1], theta)
        #     localized = True
            
        #     # Reduce the number of particles to 100 for pose tracking
        #     self._particles = self._particles[:100]
        # else:
        #     # If there are multiple clusters, reduce the number of particles
        #     # based on the number of clusters to speed up computation
        #     self._particles = self._particles[:max(100, len(self._particles) // n_clusters)]
        pose = (float("inf"), float("inf"), float("inf"))
        localized = False

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1
        # TODO: 2.5. Complete the function body with your code (i.e., replace the pass statement).

        for i, (x, y, theta) in enumerate(self._particles):
            # Add Gaussian noise to velocities
            v_noisy = v + np.random.normal(0, self._sigma_v)
            w_noisy = w + np.random.normal(0, self._sigma_w)

            # Compute new positions with noisy velocities
            x_new = x + v_noisy * self._dt * np.cos(float(theta))
            y_new = y + v_noisy * self._dt * np.sin(float(theta))
            theta_new = theta + w_noisy * self._dt

            # Normalize theta to the range [0, 2π)
            theta_new %= 2 * np.pi

            # Check for collisions with the map
            intersection, _ = self._map.check_collision([(x, y), (x_new, y_new)])
            if intersection:
                # Adjust particle to stay at the intersection point
                x_new, y_new = intersection
            
            # Update the particle
            self._particles[i] = (x_new, y_new, theta_new)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 2.9. Complete the function body with your code (i.e., replace the pass statement).
        # Compute weights for each particle
        weights = [self._measurement_probability(measurements, p) for p in self._particles] #AQUI ESTÁ EL ERROR
        self._logger.warn(f"Weights: {weights}")

        # Normalize weights to sum to 1. No haría falta, encima si es muy pequeño puede dar problemas
        weights /= np.sum(weights) # la función de random obliga a hacerlo
        

        # Resample particles using numpy's choice function
        # With the "p" argument, higher weight = higher selection probability, it's not completely random
        indices = np.random.choice(len(self._particles), size=self._particle_count, p=weights, replace=True)

        # Update the particle set with the resampled particles
        self._particles = self._particles[indices]


    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 2.4. Complete the missing function body with your code.
        # Use bounds from the Map object
        # x_min, y_min, x_max, y_max = self._map.bounds()

        # # Possible orientations (in radians)
        # orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        # i = 0  # Particle counter
        # while i < particle_count:
        #     # Sample random (x, y) within map bounds
        #     x = np.random.uniform(x_min, x_max)
        #     y = np.random.uniform(y_min, y_max)

        #     # Sample a random orientation from the allowed set
        #     theta = np.random.choice(orientations)

        #     # Check if the particle is within a valid area
        #     if not self._map.contains((x, y)):
        #         continue  # Skip if the particle is in an obstacle or outside the map

        #     # Store the valid particle
        #     particles[i] = (x, y, theta)
        #     i += 1  # Increment only when a valid particle is found

        particles[0] = (0, -1.5, 0.5 * math.pi)
        particles[1] = (-1, 1.5, 0.5 * math.pi)

        return particles

    @staticmethod
    def _init_sensor_polar_coordinates(
        sensors: list[tuple[float, float, float]],
    ) -> tuple[list[float], list[float]]:
        """Converts the sensors' poses to polar coordinates wrt to the robot's coordinate frame.

        Args:
            sensors: Robot sensors location and orientation (x, y, theta) [m, m, rad].

        Return:
            ds: List of magnitudes [m].
            phi: List of angles [rad].

        """
        ds = [math.sqrt(sensor[0] ** 2 + sensor[1] ** 2) for sensor in sensors]
        phi = [math.atan2(sensor[1], sensor[0]) for sensor in sensors]

        return ds, phi

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; inf if a sensor is out of range.

        """

        rays: list[list[tuple[float, float]]] = self._sensor_rays(particle)
        z_hat: list[float] = []

        # TODO: 2.6. Complete the missing function body with your code.
        # We first verify if a ray strikes somewhere in the map
        rays: list[list[tuple[float, float]]] = self._sensor_rays(particle)
        z_hat: list[float] = []

        for ray in rays:
            _, distance = self._map.check_collision(ray, compute_distance=True)
            z_hat.append(distance)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 2.7. Complete the function body (i.e., replace the code below).
        # Check for a valid standard deviation
        if sigma <= 0:
            raise ValueError("Standard deviation (sigma) must be positive.")

        # Calculate the Gaussian value
        coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
        exponent = -0.5 * ((x - mu) / sigma) ** 2 #AQUI ESTA EL ERROR
        return coefficient * math.exp(exponent)

    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        1.25 times the sensor range to perform the computation. This value has experimentally been
        proven valid to deal with missing measurements. Nevertheless, it might not be the optimal
        replacement value.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0

        # TODO: 2.8. Complete the missing function body with your code.
        # Predictions for a specific particle
        predicted_measurements = self._sense(particle)

        self._logger.warn(f"Predicted measurements: {predicted_measurements}")
        self._logger.warn(f"Real measurements: {measurements}")

        # We go through each measurement and its corresponding prediction and compare them
        for z, z_hat in zip(measurements, predicted_measurements):
            if z == float("inf"):  
                # If sensor doesn't detect anything, we replace for 1.25 times the sensor range
                z = 1.25 * self._sensor_range
            if z_hat == float("inf"):
                # We do the same if the prediction is infinite
                z_hat = 1.25 * self._sensor_range

            probability *= self._gaussian(mu=z_hat, sigma=self._sigma_z, x=z)

        return probability

    def _sensor_rays(self, particle: tuple[float, float, float]) -> list[list[tuple[float, float]]]:
        """Determines the simulated sensor ray segments for a given particle.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: Ray segments. Format:
                 [[(x0_begin, y0_begin), (x0_end, y0_end)],
                  [(x1_begin, y1_begin), (x1_end, y1_end)],
                  ...]

        """
        x = particle[0]
        y = particle[1]
        theta = particle[2]

        # Convert sensors to world coordinates
        xw = [x + ds * math.cos(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        yw = [y + ds * math.sin(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        tw = [sensor[2] for sensor in self._sensors]

        rays = []

        for xs, ys, ts in zip(xw, yw, tw):
            x_end = xs + self._sensor_range * math.cos(theta + ts)
            y_end = ys + self._sensor_range * math.sin(theta + ts)
            rays.append([(xs, ys), (x_end, y_end)])

        return rays
