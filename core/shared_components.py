"""
Shared Core Components for Intelligent Weapon Detection System

Eliminates code duplication between main.py and extramain.py by providing
centralized, reusable components for camera management, buffering, and alerts.
"""

import cv2
import numpy as np
import threading
import time
import json
import queue
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from utils.logging_system import get_logger
from utils.memory_manager import get_memory_manager, FrameBuffer
from utils.error_handling import handle_errors, CameraError, StorageError


@dataclass
class CameraInfo:
    """Camera information structure"""
    id: str
    name: str
    address: str
    city: str
    lat: float
    lng: float
    type: str
    index: int
    working: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FrameData:
    """Frame data structure"""
    frame: np.ndarray
    camera_id: str
    camera_info: CameraInfo
    timestamp: float
    fps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding frame for JSON)"""
        return {
            "camera_id": self.camera_id,
            "camera_info": self.camera_info.to_dict(),
            "timestamp": self.timestamp,
            "fps": self.fps,
            "frame_shape": self.frame.shape if self.frame is not None else None
        }


class CameraManager:
    """
    Centralized camera management system
    
    Features:
    - Multi-camera support with threading
    - Automatic camera detection and validation
    - Camera health monitoring
    - Graceful camera failure handling
    """
    
    def __init__(self, cameras: List[CameraInfo]):
        self.cameras = cameras
        self.working_cameras = []
        self.camera_captures: Dict[str, cv2.VideoCapture] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.running = False
        self.logger = get_logger()
        
        # Performance tracking
        self.camera_stats: Dict[str, Dict[str, Any]] = {}
        self.last_frame_time: Dict[str, float] = {}
        
        # Thread safety
        self.camera_lock = threading.RLock()
        
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize all cameras"""
        self.logger.info(f"Initializing {len(self.cameras)} cameras", extra={
            "component": "CameraManager",
            "total_cameras": len(self.cameras)
        })
        
        for camera in self.cameras:
            try:
                capture = cv2.VideoCapture(camera.index)
                if capture.isOpened():
                    # Test camera
                    ret, frame = capture.read()
                    if ret and frame is not None:
                        self.camera_captures[camera.id] = capture
                        self.frame_queues[camera.id] = queue.Queue(maxsize=10)
                        self.camera_stats[camera.id] = {
                            "frames_read": 0,
                            "errors": 0,
                            "last_frame_time": time.time(),
                            "fps": 0.0,
                            "resolution": (frame.shape[1], frame.shape[0])
                        }
                        self.working_cameras.append(camera)
                        
                        self.logger.info(f"Camera initialized: {camera.name}", extra={
                            "component": "CameraManager",
                            "camera_id": camera.id,
                            "camera_name": camera.name,
                            "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                        })
                    else:
                        capture.release()
                        self.logger.warning(f"Camera test failed: {camera.name}", extra={
                            "component": "CameraManager",
                            "camera_id": camera.id
                        })
                else:
                    self.logger.warning(f"Camera not accessible: {camera.name}", extra={
                        "component": "CameraManager",
                        "camera_id": camera.id
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize camera {camera.name}", exception=e, extra={
                    "component": "CameraManager",
                    "camera_id": camera.id
                })
        
        self.logger.info(f"Successfully initialized {len(self.working_cameras)}/{len(self.cameras)} cameras", extra={
            "component": "CameraManager",
            "working_cameras": len(self.working_cameras)
        })
    
    def start_all_cameras(self):
        """Start all camera threads"""
        if self.running:
            return
        
        self.running = True
        
        for camera in self.working_cameras:
            thread = threading.Thread(
                target=self._camera_reader_thread,
                args=(camera,),
                daemon=True
            )
            self.camera_threads[camera.id] = thread
            thread.start()
            
            self.logger.info(f"Started camera thread: {camera.name}", extra={
                "component": "CameraManager",
                "camera_id": camera.id
            })
        
        self.logger.info(f"All camera threads started ({len(self.camera_threads)})", extra={
            "component": "CameraManager"
        })
    
    def stop_all_cameras(self):
        """Stop all camera threads"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        # Release cameras
        for camera_id, capture in self.camera_captures.items():
            capture.release()
            
        self.logger.info("All cameras stopped", extra={
            "component": "CameraManager"
        })
    
    @handle_errors(default_return=None, swallow_errors=True)
    def _camera_reader_thread(self, camera: CameraInfo):
        """Camera reader thread for continuous frame capture"""
        capture = self.camera_captures[camera.id]
        frame_queue = self.frame_queues[camera.id]
        
        while self.running:
            try:
                ret, frame = capture.read()
                current_time = time.time()
                
                if ret and frame is not None:
                    # Update stats
                    stats = self.camera_stats[camera.id]
                    stats["frames_read"] += 1
                    stats["last_frame_time"] = current_time
                    
                    # Calculate FPS
                    if "last_frame_time" in stats and stats["last_frame_time"] > 0:
                        time_diff = current_time - stats["last_frame_time"]
                        if time_diff > 0:
                            stats["fps"] = 1.0 / time_diff
                    
                    # Create frame data
                    frame_data = FrameData(
                        frame=frame,
                        camera_id=camera.id,
                        camera_info=camera,
                        timestamp=current_time,
                        fps=stats["fps"]
                    )
                    
                    # Add to queue (non-blocking)
                    try:
                        frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                
                else:
                    stats["errors"] += 1
                    self.logger.warning(f"Failed to read frame from {camera.name}", extra={
                        "component": "CameraManager",
                        "camera_id": camera.id,
                        "total_errors": stats["errors"]
                    })
                    
                    # Attempt camera recovery
                    if stats["errors"] > 10:
                        self._recover_camera(camera)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Camera thread error for {camera.name}", exception=e, extra={
                    "component": "CameraManager",
                    "camera_id": camera.id
                })
                stats["errors"] += 1
                time.sleep(0.1)
    
    def _recover_camera(self, camera: CameraInfo):
        """Attempt to recover failed camera"""
        try:
            capture = self.camera_captures[camera.id]
            capture.release()
            
            # Reinitialize camera
            new_capture = cv2.VideoCapture(camera.index)
            if new_capture.isOpened():
                ret, frame = new_capture.read()
                if ret and frame is not None:
                    self.camera_captures[camera.id] = new_capture
                    self.camera_stats[camera.id]["errors"] = 0
                    
                    self.logger.info(f"Camera recovered: {camera.name}", extra={
                        "component": "CameraManager",
                        "camera_id": camera.id
                    })
                else:
                    new_capture.release()
            else:
                self.logger.error(f"Camera recovery failed: {camera.name}", extra={
                    "component": "CameraManager",
                    "camera_id": camera.id
                })
                
        except Exception as e:
            self.logger.error(f"Camera recovery error for {camera.name}", exception=e, extra={
                "component": "CameraManager",
                "camera_id": camera.id
            })
    
    def get_frames(self) -> List[FrameData]:
        """Get latest frames from all cameras"""
        frames = []
        
        for camera_id, frame_queue in self.frame_queues.items():
            try:
                # Get latest frame (non-blocking)
                while not frame_queue.empty():
                    frame_data = frame_queue.get_nowait()
                    frames.append(frame_data)
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error getting frame from camera {camera_id}", exception=e, extra={
                    "component": "CameraManager",
                    "camera_id": camera_id
                })
        
        return frames
    
    def get_camera_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cameras"""
        with self.camera_lock:
            return self.camera_stats.copy()
    
    def is_camera_healthy(self, camera_id: str) -> bool:
        """Check if camera is healthy"""
        if camera_id not in self.camera_stats:
            return False
        
        stats = self.camera_stats[camera_id]
        current_time = time.time()
        
        # Check if camera is producing frames
        time_since_last_frame = current_time - stats["last_frame_time"]
        error_rate = stats["errors"] / max(stats["frames_read"], 1)
        
        return (time_since_last_frame < 5.0 and  # Last frame within 5 seconds
                error_rate < 0.1 and  # Error rate less than 10%
                stats["frames_read"] > 0)  # At least one frame read


class BufferManager:
    """
    Centralized buffer management system
    
    Features:
    - Adaptive memory management
    - Per-camera frame buffering
    - Automatic cleanup
    - Performance optimization
    """
    
    def __init__(self, max_memory_mb: float = 1024):
        self.memory_manager = get_memory_manager()
        self.logger = get_logger()
        self.max_memory_mb = max_memory_mb
        
        # Pre-buffering support
        self.prebuffer_target = 300  # 10 seconds at 30fps
        self.prebuffer_complete = False
        
        self.logger.info("BufferManager initialized", extra={
            "component": "BufferManager",
            "max_memory_mb": max_memory_mb
        })
    
    def add_frame_to_buffer(self, frame: np.ndarray, camera_id: str, timestamp: float = None) -> bool:
        """Add frame to camera buffer"""
        return self.memory_manager.add_frame(camera_id, frame, timestamp)
    
    def get_buffer_size(self, camera_id: str = None) -> int:
        """Get buffer size for camera or total"""
        if camera_id:
            buffer = self.memory_manager.get_buffer(camera_id)
            if buffer:
                stats = buffer.get_stats()
                return stats.frame_count
            return 0
        else:
            total_frames = 0
            buffer_stats = self.memory_manager.get_buffer_stats()
            for stats in buffer_stats.values():
                total_frames += stats.frame_count
            return total_frames
    
    def get_recent_frames(self, camera_id: str, count: int = 10) -> List[np.ndarray]:
        """Get recent frames from camera buffer"""
        buffer = self.memory_manager.get_buffer(camera_id)
        if buffer:
            return buffer.get_recent_frames(count)
        return []
    
    def prebuffer_frames(self, camera_manager: CameraManager, target_frames: int = 300):
        """Pre-buffer frames for all cameras"""
        self.logger.info(f"Starting pre-buffering ({target_frames} frames)", extra={
            "component": "BufferManager",
            "target_frames": target_frames
        })
        
        prebuffer_count = 0
        start_time = time.time()
        
        while prebuffer_count < target_frames and prebuffer_count < 600:  # Safety limit
            frames = camera_manager.get_frames()
            
            for frame_data in frames:
                if frame_data.frame is not None:
                    if self.add_frame_to_buffer(frame_data.frame, frame_data.camera_id, frame_data.timestamp):
                        prebuffer_count += 1
            
            if prebuffer_count < target_frames:
                time.sleep(0.01)
        
        prebuffer_time = time.time() - start_time
        
        # Log per-camera buffer sizes
        for camera in camera_manager.working_cameras:
            buffer_size = self.get_buffer_size(camera.id)
            self.logger.info(f"Pre-buffer complete for {camera.name}: {buffer_size} frames", extra={
                "component": "BufferManager",
                "camera_id": camera.id,
                "camera_name": camera.name,
                "buffer_size": buffer_size
            })
        
        self.prebuffer_complete = True
        
        self.logger.info(f"Pre-buffering complete: {prebuffer_count} frames in {prebuffer_time:.2f}s", extra={
            "component": "BufferManager",
            "total_frames": prebuffer_count,
            "duration_seconds": prebuffer_time
        })
    
    def cleanup_old_buffers(self, age_seconds: int = 60):
        """Cleanup old frame buffers"""
        self.memory_manager.cleanup_all()
        
        self.logger.info("Buffer cleanup completed", extra={
            "component": "BufferManager",
            "age_seconds": age_seconds
        })


class AlertManager:
    """
    Centralized alert management system
    
    Features:
    - Multi-channel alert delivery
    - Alert rate limiting
    - Alert prioritization
    - Alert history tracking
    """
    
    def __init__(self, alert_cooldown_seconds: float = 5.0):
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.last_alert_time: Dict[str, float] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        self.logger = get_logger()
        
        # Thread safety
        self.alert_lock = threading.RLock()
        
        # Statistics
        self.alert_count = 0
        self.alerts_by_type: Dict[str, int] = {}
        
        self.logger.info("AlertManager initialized", extra={
            "component": "AlertManager",
            "cooldown_seconds": alert_cooldown_seconds
        })
    
    def register_callback(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
        
        self.logger.info(f"Alert callback registered: {callback.__name__}", extra={
            "component": "AlertManager",
            "callback_name": callback.__name__
        })
    
    def create_alert(self, camera_info: CameraInfo, detection_data: Dict[str, Any], 
                    frame: np.ndarray = None) -> Dict[str, Any]:
        """Create structured alert"""
        alert_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Determine alert type and priority
        alert_type = self._determine_alert_type(detection_data)
        priority = self._determine_priority(alert_type, detection_data)
        
        alert = {
            "id": alert_id,
            "timestamp": timestamp,
            "camera_id": camera_info.id,
            "camera_name": camera_info.name,
            "camera_location": camera_info.address,
            "camera_city": camera_info.city,
            "coordinates": {
                "lat": camera_info.lat,
                "lng": camera_info.lng
            },
            "alert_type": alert_type,
            "priority": priority,
            "detection_data": detection_data,
            "frame_available": frame is not None,
            "status": "active"
        }
        
        return alert
    
    def _determine_alert_type(self, detection_data: Dict[str, Any]) -> str:
        """Determine alert type from detection data"""
        if detection_data.get("explosion_conf", 0) > 0.3:
            return "explosion"
        elif detection_data.get("grenade_conf", 0) > 0.3:
            return "grenade"
        elif detection_data.get("gun_conf", 0) > 0.3:
            return "gun"
        elif detection_data.get("knife_conf", 0) > 0.3:
            return "knife"
        elif detection_data.get("violence_detected", False):
            return "violence"
        elif detection_data.get("fire_conf", 0) > 0.3:
            return "fire"
        elif detection_data.get("smoke_conf", 0) > 0.3:
            return "smoke"
        else:
            return "threat"
    
    def _determine_priority(self, alert_type: str, detection_data: Dict[str, Any]) -> str:
        """Determine alert priority"""
        high_priority_types = ["explosion", "grenade", "fire"]
        medium_priority_types = ["gun", "violence"]
        
        if alert_type in high_priority_types:
            return "critical"
        elif alert_type in medium_priority_types:
            return "high"
        else:
            return "medium"
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert with cooldown check"""
        camera_id = alert["camera_id"]
        current_time = time.time()
        
        with self.alert_lock:
            # Check cooldown
            last_time = self.last_alert_time.get(camera_id, 0)
            if current_time - last_time < self.alert_cooldown_seconds:
                return False
            
            # Update last alert time
            self.last_alert_time[camera_id] = current_time
            
            # Update statistics
            self.alert_count += 1
            alert_type = alert["alert_type"]
            self.alerts_by_type[alert_type] = self.alerts_by_type.get(alert_type, 0) + 1
            
            # Add to history
            alert["sent_timestamp"] = current_time
            self.alert_history.append(alert)
            
            # Maintain history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Send to callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {callback.__name__}", exception=e, extra={
                        "component": "AlertManager",
                        "callback_name": callback.__name__
                    })
            
            self.logger.info(f"Alert sent: {alert_type} from {alert['camera_name']}", extra={
                "component": "AlertManager",
                "alert_id": alert["id"],
                "alert_type": alert_type,
                "camera_id": camera_id,
                "priority": alert["priority"]
            })
            
            return True
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self.alert_lock:
            return {
                "total_alerts": self.alert_count,
                "alerts_by_type": dict(self.alerts_by_type),
                "recent_alerts": len([a for a in self.alert_history 
                                   if time.time() - a.get("sent_timestamp", 0) < 3600]),
                "active_cameras": len(self.last_alert_time)
            }
    
    def get_recent_alerts(self, count: int = 10, alert_type: str = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        with self.alert_lock:
            alerts = self.alert_history.copy()
            
            if alert_type:
                alerts = [a for a in alerts if a["alert_type"] == alert_type]
            
            return sorted(alerts, key=lambda x: x.get("sent_timestamp", 0), reverse=True)[:count]


class SystemCore:
    """
    Main system core that combines all shared components
    
    This is the central hub that eliminates code duplication between main.py
    and extramain.py by providing unified access to all system components.
    """
    
    def __init__(self, cameras: List[CameraInfo], max_memory_mb: float = 1024):
        self.logger = get_logger()
        
        # Initialize components
        self.camera_manager = CameraManager(cameras)
        self.buffer_manager = BufferManager(max_memory_mb)
        self.alert_manager = AlertManager()
        
        # System state
        self.running = False
        self.start_time = time.time()
        
        self.logger.info("SystemCore initialized", extra={
            "component": "SystemCore",
            "cameras": len(cameras),
            "max_memory_mb": max_memory_mb
        })
    
    def start(self):
        """Start the system"""
        self.logger.info("Starting SystemCore", extra={
            "component": "SystemCore"
        })
        
        # Start cameras
        self.camera_manager.start_all_cameras()
        
        # Pre-buffer frames
        self.buffer_manager.prebuffer_frames(self.camera_manager)
        
        self.running = True
        
        self.logger.info("SystemCore started successfully", extra={
            "component": "SystemCore"
        })
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping SystemCore", extra={
            "component": "SystemCore"
        })
        
        self.running = False
        
        # Stop cameras
        self.camera_manager.stop_all_cameras()
        
        # Cleanup buffers
        self.buffer_manager.cleanup_old_buffers()
        
        self.logger.info("SystemCore stopped", extra={
            "component": "SystemCore"
        })
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = time.time() - self.start_time
        
        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "cameras": {
                "total": len(self.camera_manager.cameras),
                "working": len(self.camera_manager.working_cameras),
                "healthy": sum(1 for cam in self.camera_manager.working_cameras 
                              if self.camera_manager.is_camera_healthy(cam.id))
            },
            "buffers": {
                "total_frames": self.buffer_manager.get_buffer_size(),
                "prebuffer_complete": self.buffer_manager.prebuffer_complete
            },
            "alerts": self.alert_manager.get_alert_statistics(),
            "memory": self.buffer_manager.memory_manager.get_system_memory_stats()
        }
    
    def process_frame(self, frame_data: FrameData, detection_system) -> Dict[str, Any]:
        """Process frame through detection system"""
        try:
            # Run detection
            detections, additional_data = detection_system.detect_objects(frame_data.frame)
            
            # Process detections
            processed_detections = detection_system.process_detections(detections, frame_data.frame)
            
            # Create alert if needed
            alert = None
            if detections:
                alert = self.alert_manager.create_alert(
                    frame_data.camera_info, 
                    processed_detections[0] if processed_detections else {},
                    frame_data.frame
                )
                self.alert_manager.send_alert(alert)
            
            return {
                "frame_data": frame_data,
                "detections": detections,
                "processed_detections": processed_detections,
                "alert": alert,
                "additional_data": additional_data
            }
            
        except Exception as e:
            self.logger.error("Frame processing error", exception=e, extra={
                "component": "SystemCore",
                "camera_id": frame_data.camera_id
            })
            return {
                "frame_data": frame_data,
                "error": str(e),
                "detections": [],
                "processed_detections": [],
                "alert": None
            }
