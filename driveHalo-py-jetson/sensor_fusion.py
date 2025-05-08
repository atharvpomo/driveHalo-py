import numpy as np
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from scipy.spatial.transform import Rotation
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from utils import measure_execution_time


class SensorFusion:
    """Sensor fusion implementation using Unscented Kalman Filter"""

    def __init__(self, dt: float = 0.05):
        self.logger = logging.getLogger("SensorFusion")
        self.logger.info("Initializing sensor fusion system")

        # State dimensions
        # x = [position_x, position_y, velocity_x, velocity_y, heading]
        self.state_dim = 5
        self.measurement_dim = 5

        # Initialize Unscented Kalman Filter
        # Use MerweScaledSigmaPoints for sigma point selection
        sigma_points = MerweScaledSigmaPoints(
            n=self.state_dim,
            alpha=0.1,
            beta=2.0,
            kappa=0.0
        )

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.state_dim,
            dim_z=self.measurement_dim,
            dt=dt,
            fx=self._state_transition_fn,
            hx=self._measurement_fn,
            points=sigma_points
        )

        # Initialize state [x, y, vx, vy, heading]
        self.ukf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Initial covariance matrix
        self.ukf.P = np.diag([10.0, 10.0, 5.0, 5.0, np.radians(30.0)]) ** 2

        # Process noise
        self.ukf.Q = np.diag([0.01, 0.01, 0.5, 0.5, 0.1]) ** 2

        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 1.0, 1.0, np.radians(5.0)]) ** 2

        # Timing and thread safety
        self.last_update_time = time.time()
        self.dt = dt
        self.lock = threading.RLock()

        # Lane state
        self.lane_curvature = float('inf')
        self.lane_width = 3.5  # meters

        # Sensor weights
        self.imu_weight = 0.6
        self.lane_weight = 0.4

        # Add GPS state to fusion
        self.gps_position = None
        self.gps_route = []

        self.logger.info("Sensor fusion initialized")

    def _state_transition_fn(self, x, dt):
        """State transition function for UKF"""
        # x = [position_x, position_y, velocity_x, velocity_y, heading]
        heading = x[4]

        # New state calculation
        new_x = np.zeros_like(x)
        new_x[0] = x[0] + x[2] * dt  # position_x += velocity_x * dt
        new_x[1] = x[1] + x[3] * dt  # position_y += velocity_y * dt
        new_x[2] = x[2]  # velocity_x (constant)
        new_x[3] = x[3]  # velocity_y (constant)
        new_x[4] = x[4]  # heading (constant)

        return new_x

    def _measurement_fn(self, x):
        """Measurement function for UKF"""
        # In this simple case, we directly observe the state
        return x

    @measure_execution_time
    def update_from_imu(self, imu_data: Dict[str, Any]) -> None:
        """Update state using IMU data"""
        with self.lock:
            # Extract IMU data
            linear_accel = imu_data.get("linear_acceleration", np.zeros(3))
            orientation_q = imu_data.get("orientation_quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
            angular_velocity = imu_data.get("angular_velocity", np.zeros(3))

            current_time = time.time()
            dt = current_time - self.last_update_time

            # Skip if dt is too small or too large
            if dt < 0.001 or dt > 0.5:
                self.last_update_time = current_time
                return

            # Convert quaternion to euler angles
            rotation = Rotation.from_quat([
                orientation_q[1], orientation_q[2], orientation_q[3], orientation_q[0]
            ])
            euler = rotation.as_euler('xyz')
            heading = euler[2]  # yaw

            # Project acceleration into vehicle frame, then to world frame
            # Assuming simple 2D motion for now
            accel_x = linear_accel[0] * np.cos(heading) - linear_accel[1] * np.sin(heading)
            accel_y = linear_accel[0] * np.sin(heading) + linear_accel[1] * np.cos(heading)

            # Current velocity from state
            vx = self.ukf.x[2]
            vy = self.ukf.x[3]

            # Update velocity with acceleration
            vx += accel_x * dt
            vy += accel_y * dt

            # Create measurement vector [x, y, vx, vy, heading]
            # Note: IMU doesn't directly measure position, so we use the predicted positions
            z = np.array([
                self.ukf.x[0],  # Use predicted x position
                self.ukf.x[1],  # Use predicted y position
                vx,  # Updated velocity x
                vy,  # Updated velocity y
                heading  # Measured heading
            ])

            # Create measurement noise matrix R
            # High uncertainty for position (IMU doesn't measure it)
            # Lower uncertainty for velocity and heading (IMU is good at these)
            R = np.diag([100.0, 100.0, 0.5, 0.5, np.radians(2.0)]) ** 2
            self.ukf.R = R

            # Perform UKF update
            self.ukf.predict(dt=dt)
            self.ukf.update(z)

            self.last_update_time = current_time

    @measure_execution_time
    def update_from_lane_detection(self, lane_data: Dict[str, Any]) -> None:
        """Update state using lane detection data"""
        with self.lock:
            # Extract lane data
            lateral_offset = lane_data.get("lateral_offset", 0.0)
            heading_error = lane_data.get("heading_error", 0.0)
            curvature_radius = lane_data.get("curvature_radius", float('inf'))
            detection_confidence = lane_data.get("confidence", 0.0)

            # Update lane curvature
            self.lane_curvature = curvature_radius

            # Skip if confidence is too low
            if detection_confidence < 0.3:
                return

            current_time = time.time()
            dt = current_time - self.last_update_time

            # Skip if dt is too small or too large
            if dt < 0.001 or dt > 0.5:
                return

            # Current heading from state
            current_heading = self.ukf.x[4]

            # Adjust heading with heading error
            corrected_heading = current_heading - heading_error

            # Current position
            x = self.ukf.x[0]
            y = self.ukf.x[1]

            # Calculate corrected position based on lateral offset
            # This is a simplification - in reality you'd need the lane reference path
            corrected_x = x
            corrected_y = y - lateral_offset  # Assuming y-axis is lateral direction

            # Assuming velocity is maintained
            vx = self.ukf.x[2]
            vy = self.ukf.x[3]

            # Create measurement vector
            z = np.array([
                corrected_x,  # Corrected x position
                corrected_y,  # Corrected y position
                vx,  # Unchanged velocity x
                vy,  # Unchanged velocity y
                corrected_heading  # Corrected heading
            ])

            # Create measurement noise matrix R
            # Lower uncertainty for position and heading (lane detection is good at these)
            # Higher uncertainty for velocity (lane detection doesn't measure it)
            confidence_factor = max(0.3, detection_confidence)
            R = np.diag([
                1.0 / confidence_factor,
                0.5 / confidence_factor,
                5.0,
                5.0,
                np.radians(5.0) / confidence_factor
            ]) ** 2
            self.ukf.R = R

            # Perform UKF update
            self.ukf.predict(dt=dt)
            self.ukf.update(z)

            self.last_update_time = current_time

    def predict(self, dt: Optional[float] = None) -> None:
        """Predict state forward in time"""
        with self.lock:
            if dt is None:
                dt = time.time() - self.last_update_time

            # Skip if dt is too small or too large
            if dt < 0.001 or dt > 0.5:
                return

            # Predict state forward
            self.ukf.predict(dt=dt)

    def get_fused_state(self) -> Dict[str, Any]:
        """Get the current fused state estimate"""
        with self.lock:
            state = self.ukf.x.copy()
            covariance = self.ukf.P.copy()

            # Extract state components
            position_x = state[0]
            position_y = state[1]
            velocity_x = state[2]
            velocity_y = state[3]
            heading = state[4]

            # Calculate derived values
            speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
            course = np.arctan2(velocity_y, velocity_x)

            # Calculate uncertainty metrics
            position_uncertainty = np.sqrt(covariance[0, 0] + covariance[1, 1])
            velocity_uncertainty = np.sqrt(covariance[2, 2] + covariance[3, 3])
            heading_uncertainty = np.sqrt(covariance[4, 4])

            return {
                "position": np.array([position_x, position_y, 0.0]),
                "velocity": np.array([velocity_x, velocity_y, 0.0]),
                "heading": heading,
                "speed": speed,
                "course": course,
                "lane_curvature": self.lane_curvature,
                "position_uncertainty": position_uncertainty,
                "velocity_uncertainty": velocity_uncertainty,
                "heading_uncertainty": heading_uncertainty,
                "timestamp": self.last_update_time
            }

    def calculate_control_inputs(self) -> Tuple[float, float, float]:
        """Calculate control inputs based on fused state"""
        state = self.get_fused_state()

        # Extract state components
        heading = state["heading"]
        lane_curvature = state["lane_curvature"]
        position_y = state["position"][1]  # Lateral position

        # Calculate steering input based on heading and curvature
        steering_from_heading = 0.0
        if lane_curvature != float('inf') and lane_curvature != 0:
            # Calculate steering angle from curvature
            # This is a simplified model
            steering_from_curvature = np.arctan(2.5 / lane_curvature)  # 2.5m is wheelbase
        else:
            steering_from_curvature = 0.0

        # Calculate lane centering component
        # Negative position_y means we're to the left of center
        lane_centering = -0.1 * position_y

        # Combine components
        steering = steering_from_curvature + lane_centering + steering_from_heading

        # Limit steering angle
        max_steering = np.radians(35.0)  # 35 degrees max steering
        steering = max(min(steering, max_steering), -max_steering)

        return heading, position_y, lane_curvature

    def update_from_gps(self, gps_data: dict):
        """Update state using GPS data"""
        with self.lock:
            if not gps_data:
                return
                
            self.gps_position = gps_data.get("position")
            self.gps_route = gps_data.get("route", [])
            
            # Update state estimation with GPS data
            if self.gps_position:
                # Create measurement vector including GPS position
                z = np.array([
                    self.gps_position[0],  # latitude
                    self.gps_position[1],  # longitude
                    self.ukf.x[2],  # Keep existing velocity x
                    self.ukf.x[3],  # Keep existing velocity y
                    self.ukf.x[4]   # Keep existing heading
                ])
                
                # Update UKF with GPS measurement
                self.ukf.predict()
                self.ukf.update(z)
