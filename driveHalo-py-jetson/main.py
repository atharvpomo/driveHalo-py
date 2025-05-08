import asyncio
import logging
import math
import cv2
import numpy as np
import argparse
import time
import threading
from typing import Tuple, Optional
import os
import collections
# Original components
from lane_detector import LaneDetector
from steering_controller import SteeringController
from vehicle_interface import MCMController
from gearshifter import GearController, GearPositions
from utils import measure_execution_time
# New components for sensor fusion
from ros_interface import ROSInterface
from imu_processor import IMUProcessor
from pose_estimator import PoseEstimator
from sensor_fusion import SensorFusion
from vehicle_state import VehicleStateManager
from scipy.spatial.transform import Rotation
import traceback
from root_logger import initialize_logging, get_logger, get_control_logger, get_diagnostic_logger, LogWindow
from diagnostic_logger import DiagnosticLogger
from gps_path_planning import GPSPathPlanner

def configure_camera(source=0, width=640, height=480, fps=10):
    """Configure camera with explicit format selection for optimal performance on Tegra"""
    if isinstance(source, int):
        # Explicitly request MJPG format to achieve 30fps
        pipeline = (
            f"v4l2src device=/dev/video{source} io-mode=2 ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            "jpegdec ! "  # Hardware-accelerated JPEG decoder
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=2 sync=false"  # Disable sync for real-time processing
        )
        logging.info(f"Initializing camera with pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logging.warning("Failed to open with MJPG pipeline, trying fallback...")
            # Fallback pipeline with explicit format-specific caps
            pipeline = (
                f"v4l2src device=/dev/video{source} ! "
                f"video/x-raw, width={width}, height={height}, framerate={fps}/1, format=YUYV ! "
                "videoconvert ! video/x-raw, format=BGR ! "
                "appsink drop=1"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # For file inputs
        cap = cv2.VideoCapture(source)
    # Verify actual configuration
    if cap.isOpened():
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
    else:
        logging.error("Failed to open camera source")
    return cap

class IMUVisualizer:
    def __init__(self, window_size=(640, 480)):
        self.window_size = window_size
        self.imu_data_buffer = collections.deque(maxlen=200)
        self.accel_data = np.zeros((200, 3))
        self.gyro_data = np.zeros((200, 3))
        cv2.namedWindow("IMU Data", cv2.WINDOW_NORMAL)
        self.orientation_history = collections.deque(maxlen=100)
        cv2.resizeWindow("IMU Data", *window_size)
        
    def update(self, imu_data):
        self.imu_data_buffer.append(imu_data)
        self._update_plot_data()
        if "orientation_quaternion" in imu_data:
            quat = imu_data["orientation_quaternion"]
            if np.linalg.norm(quat) > 0.01:
                self.orientation_history.append(quat)
    
    def update_from_pose_estimator(self, pose_state):
        """Update visualization with pose estimator state"""
        if "orientation_quaternion" in pose_state:
            quat = pose_state["orientation_quaternion"]
            if np.linalg.norm(quat) > 0.01:  # Ensure non-zero quaternion
                # Add to history with a special flag to indicate it's from pose estimator
                # We could store this in a separate buffer if needed
                pose_state["from_pose_estimator"] = True
                self.imu_data_buffer.append(pose_state)

    def _update_plot_data(self):
        # Extract data for plotting
        for i, data in enumerate(self.imu_data_buffer):
            if i < 200:
                self.accel_data[i] = data["linear_acceleration"]
                self.gyro_data[i] = data["angular_velocity"]
                
    def render(self):
        """Render a comprehensive IMU visualization with well-organized sections"""
        # Create visualization canvas
        canvas = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        # Allocate vertical regions for different components
        header_height = 50
        data_section_height = 120
        plot_section_height = 160
        orientation_section_height = 150
        
        # Draw section separators
        separator_color = (60, 60, 60)
        cv2.line(canvas, (0, header_height), (self.window_size[0], header_height), separator_color, 1)
        cv2.line(canvas, (0, header_height + data_section_height), 
                (self.window_size[0], header_height + data_section_height), separator_color, 1)
        cv2.line(canvas, (0, header_height + data_section_height + plot_section_height), 
                (self.window_size[0], header_height + data_section_height + plot_section_height), separator_color, 1)
        cv2.line(canvas, (self.window_size[0]//2, header_height), 
                (self.window_size[0]//2, header_height + data_section_height), separator_color, 1)
        
        # 1. Render header with title
        title_color = (200, 200, 200)
        cv2.putText(canvas, "IMU Sensor Data Visualization", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
        
        # 2. Render current numerical values section
        self._render_current_values(canvas, header_height)
        
        # 3. Render time series plots section
        plot_y_start = header_height + data_section_height
        self._plot_time_series(canvas, self.accel_data, (0, 255, 0), 
                            offset=plot_y_start + 50, label="Accel (m/s²)")
        self._plot_time_series(canvas, self.gyro_data, (0, 165, 255), 
                            offset=plot_y_start + 130, label="Gyro (rad/s)")
        
        # 4. Render orientation visualization section
        orientation_y_start = header_height + data_section_height + plot_section_height
        self._render_orientation(canvas, orientation_y_start)
        
        cv2.imshow("IMU Data", canvas)

    def _render_current_values(self, canvas, y_start):
        """Display current numeric values for accelerometer and gyroscope"""
        height, width = canvas.shape[:2]
        
        # Get latest values (if available)
        if len(self.imu_data_buffer) > 0:
            latest_data = self.imu_data_buffer[-1]
            
            # Section titles
            cv2.putText(canvas, "Accel (m/s²)", (30, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "Gyro (rad/s)", (width//2 + 30, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Accelerometer values
            accel = latest_data.get("linear_acceleration", np.zeros(3))
            cv2.putText(canvas, f"X: {accel[0]:.3f}", (30, y_start + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, f"Y: {accel[1]:.3f}", (30, y_start + 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, f"Z: {accel[2]:.3f}", (30, y_start + 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Gyroscope values
            gyro = latest_data.get("angular_velocity", np.zeros(3))
            cv2.putText(canvas, f"X: {gyro[0]:.3f}", (width//2 + 30, y_start + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(canvas, f"Y: {gyro[1]:.3f}", (width//2 + 30, y_start + 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(canvas, f"Z: {gyro[2]:.3f}", (width//2 + 30, y_start + 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    def _render_orientation(self, canvas, y_start):
        """Render orientation visualization with roll, pitch, yaw values and indicators"""
        height, width = canvas.shape[:2]
        text_x = 30
        
        # Draw orientation visualization
        center_x, center_y = width // 2, y_start + 75
        radius = 60
        
        # Draw background circle
        cv2.circle(canvas, (center_x, center_y), radius, (30, 30, 30), -1)
        cv2.circle(canvas, (center_x, center_y), radius, (80, 80, 80), 2)
        
        # Get latest orientation if available
        orientation_estimated = False
        if self.orientation_history and len(self.orientation_history) > 0:
            quat = self.orientation_history[-1]
            
            # Get orientation estimation status if available
            if len(self.imu_data_buffer) > 0:
                latest_data = self.imu_data_buffer[-1]
                orientation_estimated = latest_data.get("orientation_estimated", False)
            
            # Section title with estimation indicator
            title = "Orientation (Estimated)" if orientation_estimated else "Orientation"
            cv2.putText(canvas, title, (30, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            try:
                # Create rotation matrix from quaternion - ENSURE CORRECT ORDER
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                
                # Draw roll/pitch indicator
                roll, pitch, yaw = rot.as_euler('xyz')
                
                roll_pitch_x = int(center_x + radius * 0.8 * math.sin(roll) * math.cos(pitch))
                roll_pitch_y = int(center_y - radius * 0.8 * math.sin(pitch))
                
                cv2.line(canvas, (center_x, center_y), (roll_pitch_x, roll_pitch_y), (0, 255, 0), 2)
                
                # Draw heading indicator
                heading_x = int(center_x + radius * 0.8 * math.sin(yaw))
                heading_y = int(center_y - radius * 0.8 * math.cos(yaw))
                
                cv2.line(canvas, (center_x, center_y), (heading_x, heading_y), (0, 0, 255), 2)
                
                # Draw coordinate axes indicators
                cv2.line(canvas, (center_x, center_y), (center_x, center_y - 15), (255, 255, 255), 1)  # Up indicator
                cv2.line(canvas, (center_x, center_y), (center_x + 15, center_y), (255, 255, 255), 1)  # Right indicator
                
                # Add Euler angle values with clear labeling
                roll_label = "Roll*" if orientation_estimated else "Roll"
                pitch_label = "Pitch*" if orientation_estimated else "Pitch"
                yaw_label = "Yaw*" if orientation_estimated else "Yaw"
                
                cv2.putText(canvas, f"{roll_label}: {math.degrees(roll):.1f}°", (text_x, y_start + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(canvas, f"{pitch_label}: {math.degrees(pitch):.1f}°", (text_x, y_start + 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(canvas, f"{yaw_label}: {math.degrees(yaw):.1f}°", (text_x, y_start + 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add note about estimation when applicable
                if orientation_estimated:
                    cv2.putText(canvas, "*Estimated from accel/gyro", (text_x, y_start + 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
            except Exception as e:
                # Display error message for debugging
                cv2.putText(canvas, f"Orientation error: {str(e)[:20]}", (text_x, y_start + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                self.logger.error(traceback.format_exc())
        else:
            # Indicate no orientation data available
            cv2.putText(canvas, "Orientation (No Data)", (30, y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(canvas, "No orientation data available", (text_x, y_start + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def _plot_time_series(self, canvas, data, color, offset=80, label="Data"):
        height, width = canvas.shape[:2]
        
        # Draw label
        cv2.putText(canvas, label, (10, offset-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw axes
        cv2.line(canvas, (50, offset), (width-20, offset), (50, 50, 50), 1)  # x-axis
        
        # Normalize data for plotting
        if data.shape[0] > 0:
            min_val = np.min(data) if np.min(data) != np.max(data) else -1
            max_val = np.max(data) if np.min(data) != np.max(data) else 1
            scale = 60.0 / max(0.1, max_val - min_val)
            
            # Plot each axis (x,y,z) with different brightness
            colors = [(color[0], color[1], color[2]), 
                    (color[0]//2, color[1]//2, color[2]//2),
                    (color[0]//3, color[1]//3, color[2]//3)]
            
            for axis in range(3):
                axis_label = ["X", "Y", "Z"][axis]
                cv2.putText(canvas, axis_label, (20, offset+15+(axis*15)), 
                        cv2.FONT_HERSHEY_PLAIN, 1, colors[axis], 1)
                
                # Draw time series
                points = []
                for i in range(min(data.shape[0]-1, width-70)):
                    x1 = width - 20 - i
                    x2 = width - 20 - (i+1)
                    
                    y1 = int(offset - data[data.shape[0]-1-i, axis] * scale)
                    y2 = int(offset - data[data.shape[0]-1-i-1, axis] * scale)
                    
                    cv2.line(canvas, (x1, y1), (x2, y2), colors[axis], 1)


class SensorSynchronizer:
    def __init__(self, max_time_diff=0.05):  # 50ms max difference
        self.imu_buffer = []
        self.lane_buffer = []
        self.max_time_diff = max_time_diff
        self.last_imu_timestamp = None
        self.last_lane_timestamp = None
        
    def add_imu_data(self, imu_data, timestamp):
        self.imu_buffer.append((timestamp, imu_data))
        self.last_imu_timestamp = timestamp
        self._clean_old_data()
        
    def add_lane_data(self, lane_data, timestamp):
        self.lane_buffer.append((timestamp, lane_data))
        self.last_lane_timestamp = timestamp
        self._clean_old_data()
        
    def get_synchronized_data(self):
        """Return closest matching IMU and lane data"""
        if not self.imu_buffer or not self.lane_buffer:
            return None, None
            
        # Find closest matching pairs
        lane_ts = self.lane_buffer[-1][0]
        closest_imu = min(self.imu_buffer, key=lambda x: abs(x[0] - lane_ts))
        
        # Check if within acceptable time difference
        if abs(closest_imu[0] - lane_ts) > self.max_time_diff:
            return None, None
            
        return closest_imu[1], self.lane_buffer[-1][1]
    
    def _clean_old_data(self):
        """Remove data older than 1 second"""
        current_time = time.time()
        # Clean IMU buffer
        self.imu_buffer = [(t, d) for t, d in self.imu_buffer if current_time - t < 1.0]
        # Clean lane buffer
        self.lane_buffer = [(t, d) for t, d in self.lane_buffer if current_time - t < 1.0]
    


class MultiSensorAutonomousSystem:
    """Enhanced autonomous driving system with multi-sensor fusion"""

    def __init__(self, engine_path: str, can_channel: str = 'can0', use_ros: bool = True):
        # Configure logging
        self.logger = get_logger("MultiSensorAutonomousSystem")
        self.control_logger = get_control_logger()
        self.diagnostic_logger = DiagnosticLogger(auto_log_interval=5.0)

        # keys for logger
        self.help_text = [
            "keyboard commands:",
            "q: Quit",
            "c: Toggle camera feed",
            "h: Show/hide this help",
            "d: Toggle diagnostic logging",
            "1: Toggle sensor diagnostics",
            "2: Toggle sync diagnostics",
            "3: Toggle system diagnostics",
            "4: Toggle control diagnostics", 
            "5: Toggle performance diagnostics",
            "f: Force log all diagnostics",
            "r: Restart sensor systems",
            "g: Toggle GPS diagnostics",
        ]
        self.show_help = False

        self.log_window = LogWindow(window_name="System Logs", window_size=(800,400))
        self.diagnostic_logger.set_log_window(self.log_window)
        
        try:
            # Initialize original components
            self.lane_detector = LaneDetector(engine_path)
            self.steering_controller = SteeringController()
            self.mcm = MCMController(can_channel)
            self.gear_controller = GearController(can_channel)

            # adding buffers for lane and imu data
            self.imu_data_buffer = collections.deque(maxlen=100)
            self.lane_data_buffer = collections.deque(maxlen=100)

            # Initialize new multi-sensor components
            self.vehicle_state = VehicleStateManager()
            self.sensor_fusion = SensorFusion()
            self.pose_estimator = PoseEstimator()
            self.imu_processor = IMUProcessor()

            self.imu_visualizer = IMUVisualizer()
            self.sensor_synchronizer = SensorSynchronizer(max_time_diff=0.1)
            self.display_camera_feed = False

            # ROS interface for Livox LiDAR IMU
            self.use_ros = use_ros
            if use_ros:
                self.ros_interface = ROSInterface("autonomous_vehicle")
                # Register IMU callback
                self.ros_interface.register_imu_callback(self._imu_callback)
            else:
                self.ros_interface = None

            # Threading for sensor processing
            self.should_run = True
            self.imu_thread = None
            self.fusion_thread = None

            # Initialize GPS path planner
            self.gps_planner = GPSPathPlanner(use_mock=True)  # Set to False for real GPS
            self.gps_planner.set_diagnostic_logger(self.diagnostic_logger)
            
            # Add GPS buffer
            self.gps_data_buffer = collections.deque(maxlen=100)

            self.logger.info("Multi-sensor autonomous system initialized successfully")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _imu_callback(self, imu_msg):
        """Callback for IMU data from ROS"""
        try:
            # Process IMU data
            imu_data = self.imu_processor.process_imu_data(imu_msg)
            timestamp = time.time()
            self.diagnostic_logger.add_imu_timestamp(timestamp)

            #Update visualizer
            self.imu_visualizer.update(imu_data)
            
            # Update pose estimator with IMU data
            self.pose_estimator.update_from_imu(imu_data)

            # Add to synchronizer
            self.sensor_synchronizer.add_imu_data(imu_data, timestamp)
            self.imu_data_buffer.append((timestamp, imu_msg))

            
            # Update sensor fusion
            self.sensor_fusion.update_from_imu(imu_data)

            # Update vehicle state
            self.vehicle_state.update_from_imu(imu_data)
        except Exception as e:
            self.logger.error(f"Error processing IMU data: {e}")
            self.logger.error(traceback.format_exc())

    async def _process_frame_with_fusion(self, frame: np.ndarray) -> Tuple[
    np.ndarray, Optional[Tuple[float, float, float]], np.ndarray]:
        """Process frame with sensor fusion integration"""
        start_time = time.time()
        try:
            # copying frame for visualisation
            original_frame = frame.copy()
            frame_time = time.time()

            # Detect lane and calculate curvature
            annotated_frame, radius, lateral_offset, warped_mask = self.lane_detector.detect_lane(frame)
            detection_time = time.time() - start_time
            
            self.diagnostic_logger.log_performance_metrics(
                "LaneDetection", 
                detection_time,
                {"radius": radius, "lateral_offset": lateral_offset}
            )

            bev_visualization = self.lane_detector.create_bev_visualization(warped_mask)
            display_frame = annotated_frame.copy()

            # Create lane data dictionary
            lane_data = {
                "curvature_radius": radius,
                "lateral_offset": lateral_offset,
                "heading_error": 0.0,  # Would need lane orientation calculation
                "confidence": 0.8 if radius != float('inf') else 0.0,
                "timestamp": frame_time
            }
            
            # Add to synchronizer
            self.sensor_synchronizer.add_lane_data(lane_data, frame_time)
            
            # Get synchronized data
            imu_data, synced_lane_data = self.sensor_synchronizer.get_synchronized_data()
            
            self.diagnostic_logger.add_camera_timestamp(frame_time)
            if imu_data:
                timestamp = imu_data.get("timestamp", frame_time)
                self.diagnostic_logger.add_sync_result(
                    imu_data is not None and synced_lane_data is not None, 
                    abs(timestamp - frame_time) if imu_data else 0.0
                )
            self.diagnostic_logger.log_if_needed()

            if imu_data and synced_lane_data:
                # Use synchronized data for fusion and control
                self.lane_data_buffer.append((frame_time, synced_lane_data))
                
                # adding fusion logs
                fusion_start = time.time()

                # Update fusion states with lane data
                self.pose_estimator.update_from_lane_detection(synced_lane_data)
                self.sensor_fusion.update_from_lane_detection(synced_lane_data)
                self.vehicle_state.update_from_camera(synced_lane_data)
                
                fusion_time = time.time() - fusion_start
                self.diagnostic_logger.log_performance_metrics("SensorFusion", fusion_time)

                # Get fused state for control decisions
                fused_state = self.sensor_fusion.get_fused_state()
                self.vehicle_state.update_from_sensor_fusion(fused_state)

                control_start = time.time()
                # Calculate control signals using sensor fusion
                heading, lateral_offset, curvature = self.sensor_fusion.calculate_control_inputs()

                # Calculate final control signals
                steering, throttle, brake = await self.steering_controller.coordinate_controls(curvature)

                # Update vehicle state with control signals
                self.vehicle_state.update_control_signals(steering, throttle, brake)
                control_time = time.time() - control_start
                self.diagnostic_logger.log_performance_metrics(
                    "ControlCalculation",
                    control_time,
                    {"steering": steering, "throttle": throttle, "brake": brake}
                )
            else:
                # Fallback to non-synchronized approach
                steering, throttle, brake = await self.steering_controller.coordinate_controls(radius)
            
            
            # Render IMU visualization
            self.imu_visualizer.render()

            height = frame .shape[0]
            info_text = [
                f'Curve Radius: {radius:.1f}m',
                f'Steering: {steering:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Brake: {brake:.3f}',
                f'Lateral Offset: {lateral_offset:.2f}m',
                f'Temp Release: {"Active" if self.steering_controller.is_temp_release_active else "Inactive"}',
                f'Frame Time: {(time.time() - start_time)*1000:.1f}ms'
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (20, height - 30 * (len(info_text) - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0)
                            )
                                
            # Return display, controls and bev viz
            return display_frame, (steering, throttle, brake), bev_visualization
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.logger.error(traceback.format_exc())
            blank_bev = np.zeros((self.lane_detector.warped_height, self.lane_detector.warped_width, 3), dtype=np.uint8)
            return frame, None, blank_bev
    
    async def initialize_drive_mode(self):
        """Initialize the vehicle in drive mode"""
        try:
            self.logger.info("Initializing drive mode")
            self.vehicle_state.update_system_status("INITIALIZING")

            # Clear states for both subsystems before gear change
            for subsystem_id in [0, 1]:
                self.logger.info(f"Clearing initial state for subsystem {subsystem_id}")
                self.control_logger.debug(f"Subsystem {subsystem_id} state cleared")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing

            # Apply initial brake for safety during gear change
            await self.mcm.control_request('brake', True)
            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
            await asyncio.sleep(0.2)
            self.control_logger.info(f"Initial brake force set to {self.steering_controller.permanent_brake_force}")

            # Execute gear change to Drive
            self.logger.info("Shifting to Drive mode")
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.DRIVE,
                brake_percentage=0.8,
                brake_duration=1.0
            )
            self.control_logger.info("Gear changed to DRIVE completed")

            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
            self.steering_controller.is_braking = True

            # Verify system state after gear change
            for subsystem_id in [0, 1]:
                self.logger.info(f"Verifying post-gear state for subsystem {subsystem_id}")
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.1)

            self.logger.info("Drive mode initialization completed")
            self.vehicle_state.update_system_status("ACTIVE", "")
            await asyncio.sleep(0.5)  # Allow time for final gear engagement

        except Exception as e:
            self.logger.error(f"Drive mode initialization failed: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Drive mode initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    async def safe_stop_sequence(self):
        """Execute safe stop sequence"""
        try:
            self.logger.info("Executing safe stop sequence")
            self.vehicle_state.update_system_status("SHUTDOWN", "Safe stop initiated")

            # First shift to Neutral with moderate braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.NEUTRAL,
                brake_percentage=0.4,
                brake_duration=1.5
            )
            await asyncio.sleep(0.5)  # Allow for gear disengagement

            # Then shift to Park with firm braking
            await self.gear_controller.execute_gear_change(
                gear_position=GearPositions.PARK,
                brake_percentage=0.5,
                brake_duration=2.0
            )

            # Clear states for both subsystems with proper seed-key sequence
            for subsystem_id in [0, 1]:
                await self.mcm.clear_subsystem_state(subsystem_id)
                await asyncio.sleep(0.2)  # Ensure complete processing between subsystems

            self.logger.info("Safe stop sequence completed")
            self.vehicle_state.update_system_status("SHUTDOWN", "Safe stop completed")
        except Exception as e:
            self.logger.error(f"Error during safe stop sequence: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Safe stop failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_closest(self, buffer, target_time):
        return min(buffer, key=lambda x: abs(x[0] - target_time), default=(None, None))[1]

    def _sensor_fusion_thread(self):
        """Thread for continuous sensor fusion updates"""
        try:
            while self.should_run:
                # Predict forward at regular intervals
                now = time.time()
                imu_msg = self._get_closest(self.imu_data_buffer, now)
                lane_data = self._get_closest(self.lane_data_buffer, now)
                if imu_msg and lane_data:
                    imu_data = self.imu_processor.process_imu_data(imu_msg)
                    self.sensor_fusion.update_from_imu(imu_data)
                    self.sensor_fusion.update_from_lane_detection(lane_data)
                    time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"Sensor fusion thread error: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Sensor fusion error: {e}")
            self.logger.error(traceback.format_exc())

    def start_sensor_threads(self):
        """Start threads for sensor processing"""
        try:
            # Start ROS interface if enabled
            if self.use_ros:
                self.ros_interface.start()
                # Start IMU calibration
                threading.Thread(target=self.imu_processor.start_calibration, daemon=True).start()

                # Start sensor fusion thread
                self.fusion_thread = threading.Thread(target=self._sensor_fusion_thread)
                self.fusion_thread.daemon = True
                self.fusion_thread.start()

                self.logger.info("Sensor threads started")
            else:
                # Log that we're running without sensor fusion
                self.logger.info("Running without ROS/IMU - sensor fusion disabled")
        except Exception as e:
            self.logger.error(f"Failed to start sensor threads: {e}")
            self.vehicle_state.update_system_status("ERROR", f"Sensor thread start failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def stop_sensor_threads(self):
        """Stop all sensor processing threads"""
        self.should_run = False

        # Stop ROS interface
        if self.use_ros and self.ros_interface:
            self.ros_interface.stop()

        # Wait for threads to finish
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=1.0)

        self.logger.info("Sensor threads stopped")

    def _process_gps_data(self):
        """Process GPS data and update fusion"""
        try:
            gps_data = self.gps_planner.update()
            if gps_data:
                timestamp = time.time()
                self.gps_data_buffer.append((timestamp, gps_data))
                
                # Update sensor fusion
                self.sensor_fusion.update_from_gps(gps_data)
                
                # Update vehicle state
                self.vehicle_state.update_from_gps(gps_data)
                
        except Exception as e:
            self.logger.error(f"GPS processing error: {e}")
    
    async def run(self, source=0, width=640, height=480, fps=10):
        """Main control loop with integrated sensor fusion"""
        self.logger.info(f"Opening video source: {source}")
        self.vehicle_state.update_system_status("INITIALIZING", "Starting sensor systems")

        # Start sensor threads
        self.start_sensor_threads()

        # Use custom camera configuration function
        cap = configure_camera(source, width=width, height=height, fps=fps)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            self.vehicle_state.update_system_status("ERROR", f"Failed to open video source: {source}")
            return

        try:
            # Allow time for sensor calibration
            self.logger.info("Waiting for sensor calibration...")
            for i in range(10, 0, -1):
                self.logger.info(f"Starting in {i} seconds...")
                await asyncio.sleep(1.0)

            # Initialize drive mode
            await self.initialize_drive_mode()

            # Enable all controls while maintaining brake state
            for control in ['steer', 'throttle', 'brake']:
                await self.mcm.control_request(control, True)
                self.logger.info(f"{control.capitalize()} control enabled")

            # Ensure permanent brake force is applied initially
            await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)

            cv2.namedWindow("Multi-Sensor Autonomous System", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Multi-Sensor Autonomous System", 50, 50)
            cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Bird's Eye View", 700, 50)
            cv2.resizeWindow("Bird's Eye View", 400, 500)  # Set appropriate size

            
            self.logger.info("Starting main control loop")
            self.vehicle_state.update_system_status("ACTIVE", "")

            frame_count = 0
            start_time = time.time()
            last_fps_log_time = start_time

            while cap.isOpened():
                # Process GPS data every iteration
                self._process_gps_data()
                
                success, frame = cap.read()
                if not success:
                    self.logger.error("Failed to read frame")
                    # Try to recover rather than breaking
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Log FPS every second rather than every 30 frames
                if current_time - last_fps_log_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    self.logger.info(f"Current FPS: {current_fps:.2f}")
                    last_fps_log_time = current_time

                # Process frame with sensor fusion
                annotated_frame, controls, bev_visualization = await self._process_frame_with_fusion(frame)

                if controls:
                    steering, throttle, brake = controls
                    try:
                        # Apply controls while respecting brake states
                        await self.mcm.update_setpoint('brake', brake)
                        await self.mcm.update_setpoint('steer', steering)
                        await asyncio.sleep(0.02)  # Brief delay between control updates
                        await self.mcm.update_setpoint('throttle', throttle)
                    except Exception as e:
                        self.logger.error(f"Control update failed: {e}")
                        self.vehicle_state.update_system_status("ERROR", f"Control update failed: {e}")
                        self.logger.error(traceback.format_exc())
                        # Ensure brakes are applied on error
                        await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)

                # Update display with control information
                if annotated_frame is not None:
                    fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
                    cv2.putText(annotated_frame, fps_text, (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if self.display_camera_feed:
                        cv2.imshow('Multi-Sensor Autonomous System', annotated_frame)
                        
                    if bev_visualization is not None:
                        cv2.imshow("Bird's Eye View", bev_visualization)
                
                self.log_window.render()
                        
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    self.log_window.process_key(key)
                    if key == ord('q'):
                        self.logger.info("User requested exit")
                        break
                    elif key == ord('c'):
                        self.display_camera_feed = not self.display_camera_feed
                        self.logger.info(f"Camera feed display: {'enabled' if self.display_camera_feed else 'disabled'}")
                    elif key == ord('h'):
                        self.show_help = not self.show_help
                        self.logger.info(f"Help display: {'enabled' if self.show_help else 'disabled'}")
                    elif key == ord('d'):
                        # Toggle all diagnostic logging
                        self.diagnostic_logger.toggle_auto_logging()
                    elif key == ord('1'):
                        # Toggle sensor diagnostics
                        self.log_window.add_log("system", "Toggling SENSOR daignostics")
                        self.diagnostic_logger.toggle_category("sensors")
                    elif key == ord('2'):
                        # Toggle sync diagnostics
                        self.log_window.add_log("system", "Toggling SYNC daignostics")
                        self.diagnostic_logger.toggle_category("sync")
                    elif key == ord('3'):
                        # Toggle system diagnostics
                        self.log_window.add_log("system", "Toggling system metrics daignostics")
                        self.diagnostic_logger.toggle_category("system")
                    elif key == ord('4'):
                        # Toggle control diagnostics
                        self.log_window.add_log("system", "Toggling control daignostics")
                        self.diagnostic_logger.toggle_category("control")
                    elif key == ord('5'):
                        # Toggle performance diagnostics
                        self.log_window.add_log("system", "Toggling performance daignostics")
                        self.diagnostic_logger.toggle_category("performance")
                    elif key == ord('f'):
                        # Force log all diagnostics immediately
                        self.log_window.add_log("system", "Toggling forcefull flush of all daignostics")
                        self.logger.info("Forcing diagnostic log dump")
                        self.diagnostic_logger.force_log_all()
                    elif key == ord('r'):
                        # Restart sensor systems
                        self.logger.info("Restarting sensor systems")
                        self.log_window.add_log("system", "restarting the sesnors please wait or toggle daignostics")
                        self.stop_sensor_threads()
                        await asyncio.sleep(0.5)
                        self.start_sensor_threads()
                    elif key == ord('g'):
                        self.diagnostic_logger.toggle_category("gps")
                        self.logger.info("Toggled GPS diagnostics")

                # Add a small yield to prevent CPU overutilization
                await asyncio.sleep(0.001)

        except Exception as e:
            self.logger.error(f"Runtime error: {e}", exc_info=True)
            self.vehicle_state.update_system_status("ERROR", f"Runtime error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Stop sensor threads
            self.stop_sensor_threads()

            # Always ensure brakes are applied during shutdown
            self.logger.info("Initiating shutdown sequence")
            try:
                # Sequence control shutdown
                await self.mcm.update_setpoint('throttle', 0)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('brake', self.steering_controller.permanent_brake_force)
                await asyncio.sleep(0.1)
                await self.mcm.update_setpoint('steer', 0)
                await asyncio.sleep(0.1)

                await self.safe_stop_sequence()

                # Disable all controls in reverse order
                for control in ['throttle', 'brake', 'steer']:
                    await self.mcm.control_request(control, False)
                    await asyncio.sleep(0.05)

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                self.vehicle_state.update_system_status("ERROR", f"Cleanup error: {e}")
                self.logger.error(traceback.format_exc())

            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Shutdown complete")
            self.vehicle_state.update_system_status("SHUTDOWN", "System shutdown complete")


def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Multi-Sensor Autonomous Driving System')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--can-channel', type=str, default='can0',
                        help='CAN bus channel for vehicle control')
    # Add camera configuration arguments
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width resolution')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height resolution')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera frames per second')
    # Add argument for ROS usage
    parser.add_argument('--use-ros', action='store_true',
                        help='Enable ROS integration for Livox LiDAR')
    parser.add_argument('--imu-topic', type=str, default='/livox/imu',
                        help='ROS topic for Livox IMU data')
    # Add logging configuration
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--disable-file-logs', action='store_true',
                        help='Disable logging to files (console only)')
    parser.add_argument('--diagnostic-interval', type=float, default=5.0,
                        help='Interval between automatic diagnostic logs (seconds)')

    args = parser.parse_args()

    try:
        # Initialize logging first
        log_level = getattr(logging, args.log_level)
        initialize_logging(
            log_dir=args.log_dir,
            default_level=log_level,
            enable_console=True,
            enable_file=not args.disable_file_logs
        )
        
        # Get root logger
        logger = get_logger("main")
        logger.info(f"Starting Multi-Sensor Autonomous Driving System")
        logger.info(f"Log level: {args.log_level}, Log directory: {args.log_dir}")

        # Initialize system
        system = MultiSensorAutonomousSystem(
            args.engine,
            args.can_channel,
            use_ros=args.use_ros
        )
        
        # Set diagnostic interval
        system.diagnostic_logger.auto_log_interval = args.diagnostic_interval
        logger.info(f"Diagnostic logging interval: {args.diagnostic_interval}s")

        # Parse source properly
        source = 0 if args.source == '0' else args.source
        logger.info(f"Using video source: {source}")

        # Run system
        asyncio.run(system.run(
            source=source,
            width=args.width,
            height=args.height,
            fps=args.fps
        ))
        
    except Exception as e:
        # Get logger directly in case initialization failed
        logger = logging.getLogger("main")
        logger.error(f"System failed to start: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
