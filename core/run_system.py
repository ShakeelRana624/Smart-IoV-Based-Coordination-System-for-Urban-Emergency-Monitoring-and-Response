"""
Professional Integrated Gun Detection System Launcher
Combines main.py and extramain.py functionality with professional improvements
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import cv2
import tempfile
import numpy as np
import threading
import json
import queue
import math
from collections import defaultdict
import signal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import professional systems
from utils.logging_system import setup_logging, get_logger
from utils.memory_manager import setup_memory_manager, get_memory_manager
from utils.error_handling import setup_error_handler, get_error_handler, CameraError, StorageError

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Cloudinary imports
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False

from core.integrated_gun_detection_system import IntegratedGunDetectionSystem


class FirebaseRealtimeDB:
    """Firebase Realtime Database handler with Cloudinary video upload"""
    
    # Camera configurations
    CAMERA_WAZIRABAD = {
        'id': 'CAM_WZD_001',
        'name': 'Wazirabad Security Camera',
        'address': 'Wazirabad, Pakistan',
        'city': 'Wazirabad',
        'lat': 32.245430,
        'lng': 74.163434,
        'type': 'Laptop Camera'
    }
    
    CAMERA_GUJRANWALA = {
        'id': 'CAM_GRW_001',
        'name': 'Gujranwala Security Camera',
        'address': 'Gujranwala, Pakistan',
        'city': 'Gujranwala',
        'lat': 32.221250,
        'lng': 74.172576,
        'type': 'Mobile Camera'
    }

    def __init__(self):
        self.logger = get_logger()
        self.error_handler = get_error_handler()
        self.memory_manager = get_memory_manager()
        
        self.initialized = False
        self.app = None
        self.database_url = 'https://fypiov-default-rtdb.firebaseio.com/'
        self.alert_count = 0
        
        # Cloudinary configuration
        self.cloudinary_initialized = False
        
        # Professional buffer management
        self.post_detection_frames = {}
        self.post_detection_threshold = 300  # 10 seconds post-detection
        self.detection_active = {}
        self.last_alert_time = 0
        self.video_cooldown = 5
        self.videos_uploaded = 0
        
        # Camera lookup
        self.cameras_by_id = {
            'CAM_WZD_001': self.CAMERA_WAZIRABAD,
            'CAM_GRW_001': self.CAMERA_GUJRANWALA
        }
        
        self._init()
        self._init_cloudinary()

    def _init(self):
        """Initialize Firebase Realtime Database"""
        self.logger.info("Initializing Firebase Realtime Database", extra={
            "component": "FirebaseRealtimeDB"
        })

        if not FIREBASE_AVAILABLE:
            self.logger.warning("Firebase SDK not available", extra={
                "component": "FirebaseRealtimeDB"
            })
            return

        try:
            # Find config file
            config_file = None
            for path in ["serviceAccountKey.json", "config/firebase_config.json", "firebase_config.json"]:
                if os.path.exists(path):
                    config_file = path
                    break

            if not config_file:
                self.logger.error("No Firebase config found", extra={
                    "component": "FirebaseRealtimeDB"
                })
                return

            # Initialize Firebase
            cred = credentials.Certificate(config_file)
            self.app = firebase_admin.initialize_app(cred, {
                'databaseURL': self.database_url
            }, name='realtime_alerts_app')

            # Create references
            self.alerts_ref = db.reference('alerts', app=self.app)
            self.latest_ref = db.reference('latest_alert', app=self.app)
            self.stats_ref = db.reference('system_stats', app=self.app)
            self.cameras_ref = db.reference('cameras', app=self.app)
            self.iovs_ref = db.reference('iovs', app=self.app)

            # Save cameras info
            cameras_info = {
                self.CAMERA_WAZIRABAD['id']: {
                    'camera_id': self.CAMERA_WAZIRABAD['id'],
                    'name': self.CAMERA_WAZIRABAD['name'],
                    'address': self.CAMERA_WAZIRABAD['address'],
                    'city': self.CAMERA_WAZIRABAD['city'],
                    'latitude': self.CAMERA_WAZIRABAD['lat'],
                    'longitude': self.CAMERA_WAZIRABAD['lng'],
                    'type': self.CAMERA_WAZIRABAD['type'],
                    'location': {'lat': self.CAMERA_WAZIRABAD['lat'], 'lng': self.CAMERA_WAZIRABAD['lng']},
                    'installed_at': '2026-03-07',
                    'status': 'active'
                },
                self.CAMERA_GUJRANWALA['id']: {
                    'camera_id': self.CAMERA_GUJRANWALA['id'],
                    'name': self.CAMERA_GUJRANWALA['name'],
                    'address': self.CAMERA_GUJRANWALA['address'],
                    'city': self.CAMERA_GUJRANWALA['city'],
                    'latitude': self.CAMERA_GUJRANWALA['lat'],
                    'longitude': self.CAMERA_GUJRANWALA['lng'],
                    'type': self.CAMERA_GUJRANWALA['type'],
                    'location': {'lat': self.CAMERA_GUJRANWALA['lat'], 'lng': self.CAMERA_GUJRANWALA['lng']},
                    'installed_at': '2026-03-07',
                    'status': 'active'
                }
            }
            
            self.cameras_ref.set(cameras_info)
            self.initialized = True
            
            self.logger.info("Firebase initialized successfully", extra={
                "component": "FirebaseRealtimeDB",
                "cameras_count": len(cameras_info)
            })

        except Exception as e:
            self.logger.error("Firebase initialization failed", exception=e, extra={
                "component": "FirebaseRealtimeDB"
            })

    def _init_cloudinary(self):
        """Initialize Cloudinary for video uploads"""
        if not CLOUDINARY_AVAILABLE:
            self.logger.warning("Cloudinary not available", extra={
                "component": "FirebaseRealtimeDB"
            })
            return
            
        try:
            cloudinary.config(
                cloud_name="dsnpjwaly",
                api_key="822554518314666",
                api_secret="5Mx7QjuEoe9so37yJLXM3LbJOL0",
                secure=True
            )
            
            cloudinary.api.ping()
            self.cloudinary_initialized = True
            
            self.logger.info("Cloudinary initialized successfully", extra={
                "component": "FirebaseRealtimeDB"
            })
            
        except Exception as e:
            self.logger.error("Cloudinary initialization failed", exception=e, extra={
                "component": "FirebaseRealtimeDB"
            })

    def add_frame_to_buffer(self, frame, camera_id=None):
        """Add frame to professional memory manager buffer"""
        if frame is None or camera_id is None:
            return False
            
        try:
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                return False
            
            return self.memory_manager.add_frame(camera_id, frame, time.time())
                
        except Exception as e:
            self.logger.error("Error adding frame to buffer", exception=e, extra={
                "component": "FirebaseRealtimeDB",
                "camera_id": camera_id
            })
            return False

    def get_buffer_size(self, camera_id=None):
        """Get current buffer size from professional memory manager"""
        return self.memory_manager.get_buffer_size(camera_id)

    def start_post_detection_recording(self, camera_id):
        """Start post-detection recording"""
        self.detection_active[camera_id] = True
        self.post_detection_frames[camera_id] = 0
        self.logger.info(f"Post-detection recording started for camera {camera_id}", extra={
            "component": "FirebaseRealtimeDB",
            "camera_id": camera_id
        })

    def add_post_detection_frame(self, camera_id, frame):
        """Add frame using professional memory manager"""
        if not self.detection_active.get(camera_id, False):
            return False
        
        success = self.memory_manager.add_frame(camera_id, frame, time.time())
        
        if success:
            self.post_detection_frames[camera_id] = self.post_detection_frames.get(camera_id, 0) + 1
            
            if self.post_detection_frames[camera_id] >= self.post_detection_threshold:
                self.detection_active[camera_id] = False
                self.logger.info(f"Post-detection recording complete for camera {camera_id}", extra={
                    "component": "FirebaseRealtimeDB",
                    "camera_id": camera_id,
                    "total_frames": self.post_detection_frames[camera_id]
                })
                return True
        
        return False


class CameraHandler:
    """Professional camera handler with multiple camera support"""
    
    def __init__(self, cameras_config):
        self.logger = get_logger()
        self.cameras_config = cameras_config
        self.working_cameras = []
        self.camera_captures = {}
        self.camera_threads = {}
        self.frame_queues = {}
        self.running = False
        self.camera_lock = threading.RLock()
        
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize all cameras"""
        self.logger.info(f"Initializing {len(self.cameras_config)} cameras", extra={
            "component": "CameraHandler"
        })
        
        for camera_config in self.cameras_config:
            try:
                camera_id = camera_config['id']
                camera_index = 0 if camera_id == 'CAM_WZD_001' else 1
                
                capture = cv2.VideoCapture(camera_index)
                if capture.isOpened():
                    ret, frame = capture.read()
                    if ret and frame is not None:
                        self.camera_captures[camera_id] = capture
                        self.frame_queues[camera_id] = queue.Queue(maxsize=10)
                        
                        camera_info = {
                            'id': camera_id,
                            'name': camera_config['name'],
                            'city': camera_config['city'],
                            'info': camera_config,
                            'index': camera_index
                        }
                        self.working_cameras.append(camera_info)
                        
                        self.logger.info(f"Camera initialized: {camera_config['name']}", extra={
                            "component": "CameraHandler",
                            "camera_id": camera_id
                        })
                    else:
                        capture.release()
                else:
                    self.logger.warning(f"Camera not accessible: {camera_config['name']}", extra={
                        "component": "CameraHandler",
                        "camera_id": camera_id
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize camera {camera_config['name']}", exception=e, extra={
                    "component": "CameraHandler",
                    "camera_id": camera_config['id']
                })
        
        self.logger.info(f"Successfully initialized {len(self.working_cameras)}/{len(self.cameras_config)} cameras", extra={
            "component": "CameraHandler"
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
            self.camera_threads[camera['id']] = thread
            thread.start()
            
            self.logger.info(f"Started camera thread: {camera['name']}", extra={
                "component": "CameraHandler",
                "camera_id": camera['id']
            })
    
    def stop_all_cameras(self):
        """Stop all camera threads"""
        self.running = False
        
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        for capture in self.camera_captures.values():
            capture.release()
        
        self.logger.info("All cameras stopped", extra={
            "component": "CameraHandler"
        })
    
    def _camera_reader_thread(self, camera):
        """Camera reader thread"""
        camera_id = camera['id']
        capture = self.camera_captures[camera_id]
        frame_queue = self.frame_queues[camera_id]
        
        while self.running:
            try:
                ret, frame = capture.read()
                if ret and frame is not None:
                    frame_data = {
                        'frame': frame,
                        'camera_id': camera_id,
                        'camera_info': camera,
                        'timestamp': time.time(),
                        'fps': 30.0
                    }
                    
                    try:
                        frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Camera thread error for {camera['name']}", exception=e, extra={
                    "component": "CameraHandler",
                    "camera_id": camera_id
                })
                time.sleep(0.1)
    
    def get_frames(self):
        """Get latest frames from all cameras"""
        frames = []
        
        for camera_id, frame_queue in self.frame_queues.items():
            try:
                while not frame_queue.empty():
                    frame_data = frame_queue.get_nowait()
                    frames.append(frame_data)
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error getting frame from camera {camera_id}", exception=e, extra={
                    "component": "CameraHandler",
                    "camera_id": camera_id
                })
        
        return frames
    
    def setup_cameras(self):
        """Setup cameras (compatibility method)"""
        return len(self.working_cameras) > 0


class IntegratedWeaponDetectionSystem:
    """Main integrated weapon detection system"""
    
    def __init__(self):
        # Setup professional systems
        self.logger = setup_logging()
        self.memory_manager = setup_memory_manager(max_memory_mb=2048)
        self.error_handler = setup_error_handler()
        
        # System state
        self.running = False
        self.frame_count = 0
        self.frames_added = 0
        
        # Initialize Firebase
        self.firebase_rt = FirebaseRealtimeDB()
        
        # Initialize detection system
        self.detection_system = IntegratedGunDetectionSystem()
        
        # Initialize camera handler
        cameras_config = [self.firebase_rt.CAMERA_WAZIRABAD, self.firebase_rt.CAMERA_GUJRANWALA]
        self.camera_handler = CameraHandler(cameras_config)
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("IntegratedWeaponDetectionSystem initialized", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })
        self.stop()
    
    def setup_cameras(self):
        """Setup cameras"""
        return self.camera_handler.setup_cameras()
    
    def run(self):
        """Main detection loop"""
        try:
            self.logger.info("Starting Integrated Weapon Detection System", extra={
                "component": "IntegratedWeaponDetectionSystem"
            })
            
            # Setup cameras
            if not self.setup_cameras():
                self.logger.error("No cameras available", extra={
                    "component": "IntegratedWeaponDetectionSystem"
                })
                return
            
            # Start cameras
            self.camera_handler.start_all_cameras()
            
            # Pre-buffer frames if Cloudinary is available
            if self.firebase_rt.cloudinary_initialized:
                self._prebuffer_frames()
            
            # Start detection loop
            self._run_detection_loop()
            
        except Exception as e:
            self.logger.critical("System failed to start", exception=e, extra={
                "component": "IntegratedWeaponDetectionSystem"
            })
        finally:
            self.stop()
    
    def _prebuffer_frames(self):
        """Pre-buffer frames for evidence collection"""
        self.logger.info("Starting frame pre-buffering", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })
        
        prebuffer_count = 0
        prebuffer_target = 600
        start_time = time.time()
        
        while prebuffer_count < prebuffer_target and prebuffer_count < 1200:  # Safety limit
            frames = self.camera_handler.get_frames()
            
            for frame_data in frames:
                if frame_data['frame'] is not None:
                    if self.firebase_rt.add_frame_to_buffer(frame_data['frame'], frame_data['camera_id']):
                        prebuffer_count += 1
            
            if prebuffer_count < prebuffer_target:
                time.sleep(0.01)
        
        prebuffer_time = time.time() - start_time
        
        # Log per-camera buffer sizes
        for cam in self.camera_handler.working_cameras:
            cam_id = cam['id']
            buf_size = self.firebase_rt.get_buffer_size(cam_id)
            self.logger.info(f"Pre-buffer complete for {cam['name']}: {buf_size} frames", extra={
                "component": "IntegratedWeaponDetectionSystem",
                "camera_id": cam_id,
                "buffer_size": buf_size
            })
        
        self.logger.info(f"Pre-buffering complete: {prebuffer_count} frames in {prebuffer_time:.2f}s", extra={
            "component": "IntegratedWeaponDetectionSystem",
            "total_frames": prebuffer_count,
            "duration_seconds": prebuffer_time
        })
        
        # Wait for buffer stabilization
        time.sleep(5)
    
    def _run_detection_loop(self):
        """Main detection processing loop"""
        self.logger.info("Starting weapon detection loop", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })
        
        # Print camera info
        self._print_camera_info()
        
        self.running = True
        last_status_time = time.time()
        status_interval = 60  # Log status every minute
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get frames from all cameras
                frames_data = self.camera_handler.get_frames()
                
                # Process each frame
                for frame_data in frames_data:
                    frame = frame_data['frame']
                    camera_id = frame_data['camera_id']
                    camera_info = frame_data['camera_info']
                    
                    if frame is not None:
                        self.frame_count += 1
                        
                        # Add to Firebase buffer if available
                        if self.firebase_rt.cloudinary_initialized:
                            self.firebase_rt.add_post_detection_frame(camera_id, frame)
                            self.frames_added += 1
                        
                        # Process through detection system
                        try:
                            detections, fire_smoke = self.detection_system.detect_objects(frame)
                            results = self.detection_system.process_detections(detections, frame)
                            
                            # Handle detection callback
                            if detections:
                                self._handle_detection(detections, camera_info, frame)
                            
                        except Exception as e:
                            self.logger.error("Detection processing error", exception=e, extra={
                                "component": "IntegratedWeaponDetectionSystem",
                                "camera_id": camera_id
                            })
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    self._log_system_status()
                    last_status_time = current_time
                
                # Control frame rate
                elapsed = time.time() - loop_start
                if elapsed < 0.033:  # Target ~30 FPS
                    time.sleep(0.033 - elapsed)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received", extra={
                "component": "IntegratedWeaponDetectionSystem"
            })
        except Exception as e:
            self.logger.error("Error in detection loop", exception=e, extra={
                "component": "IntegratedWeaponDetectionSystem"
            })
    
    def _print_camera_info(self):
        """Print camera information"""
        print("=" * 60)
        print("📍 WORKING CAMERAS:")
        for cam in self.camera_handler.working_cameras:
            print(f"   ✅ {cam['name']} - {cam['city']}")
            print(f"      Coordinates: {cam['info']['lat']}, {cam['info']['lng']}")
            print(f"      Range: 3km in {cam['city']} area")
        print("=" * 60)
        print("🚗 ALERTS TO: IOVs within 3km of detecting camera's city")
        print("=" * 60)
        print("Press 'q' to quit\n")
    
    def _handle_detection(self, detections, camera_info, frame):
        """Handle detection events"""
        # Extract weapon detections
        weapon_detections = []
        for detection in detections:
            meta = detection.get('meta', {})
            det_class = meta.get('class_name', '').lower()
            
            confidence = (
                detection.get('gun_conf', 0) or
                detection.get('knife_conf', 0) or
                detection.get('explosion_conf', 0) or
                detection.get('grenade_conf', 0) or
                meta.get('raw_confidence', 0)
            )
            
            # Filter for weapons
            ignore_classes = ['person', 'cell phone', 'bottle', 'cup', 'laptop']
            if any(ignore in det_class for ignore in ignore_classes):
                continue
            
            weapon_keywords = ['gun', 'pistol', 'rifle', 'shotgun', 'weapon', 'knife', 'blade']
            if any(kw in det_class for kw in weapon_keywords):
                if confidence >= 0.3:
                    weapon_detections.append({
                        'type': det_class,
                        'confidence': confidence,
                        'bbox': detection.get('bbox', []),
                        'detection': detection
                    })
        
        # Send alert if weapons detected
        if weapon_detections and self.firebase_rt.initialized:
            self._send_alert(weapon_detections, camera_info, frame)
    
    def _send_alert(self, weapon_detections, camera_info, frame):
        """Send alert to Firebase"""
        try:
            current_time = time.time()
            
            # Cooldown check
            if current_time - self.firebase_rt.last_alert_time < self.firebase_rt.video_cooldown:
                return
            
            self.firebase_rt.last_alert_time = current_time
            
            # Create alert
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'camera_id': camera_info['id'],
                'camera_name': camera_info['name'],
                'camera_location': camera_info['info']['address'],
                'camera_city': camera_info['info']['city'],
                'coordinates': {
                    'lat': camera_info['info']['lat'],
                    'lng': camera_info['info']['lng']
                },
                'detections': weapon_detections,
                'alert_type': 'weapon_detection',
                'confidence': max(w['confidence'] for w in weapon_detections),
                'status': 'active'
            }
            
            # Send to Firebase
            if self.firebase_rt.initialized:
                self.firebase_rt.alerts_ref.push(alert_data)
                self.firebase_rt.latest_ref.set(alert_data)
                self.firebase_rt.alert_count += 1
                
                self.logger.info(f"Alert sent: {weapon_detections[0]['type']} from {camera_info['name']}", extra={
                    "component": "IntegratedWeaponDetectionSystem",
                    "camera_id": camera_info['id'],
                    "alert_type": "weapon_detection",
                    "confidence": alert_data['confidence']
                })
                
        except Exception as e:
            self.logger.error("Failed to send alert", exception=e, extra={
                "component": "IntegratedWeaponDetectionSystem",
                "camera_id": camera_info['id']
            })
    
    def _log_system_status(self):
        """Log system status"""
        try:
            memory_stats = self.memory_manager.get_system_memory_stats()
            error_stats = self.error_handler.get_error_statistics()
            
            self.logger.info("System status update", extra={
                "component": "IntegratedWeaponDetectionSystem",
                "frame_count": self.frame_count,
                "frames_added": self.frames_added,
                "memory_stats": memory_stats.__dict__,
                "error_stats": error_stats
            })
            
        except Exception as e:
            self.logger.error("Error logging system status", exception=e, extra={
                "component": "IntegratedWeaponDetectionSystem"
            })
    
    def stop(self):
        """Stop the system"""
        if not self.running:
            return
        
        self.logger.info("Stopping Integrated Weapon Detection System", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })
        
        self.running = False
        
        # Stop cameras
        self.camera_handler.stop_all_cameras()
        
        # Final status log
        self._log_system_status()
        
        self.logger.info("System stopped successfully", extra={
            "component": "IntegratedWeaponDetectionSystem"
        })


def main():
    """Main entry point"""
    print("=" * 80)
    print("🚀 PROFESSIONAL INTEGRATED WEAPON DETECTION SYSTEM")
    print("Enhanced with Logging, Memory Management & Error Handling")
    print("=" * 80)
    
    # Check model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        print("Please ensure best.pt is in models/ directory")
        return
    
    # Check dependencies
    try:
        import cv2
        import ultralytics
        import numpy as np
        print("✅ Dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return
    
    # Check professional dependencies
    try:
        import psutil
        print("✅ Professional dependencies verified")
    except ImportError as e:
        print(f"⚠️ Missing professional dependency: {e}")
        print("Some features may be limited")
    
    try:
        print("🚀 Starting professional detection system...")
        system = IntegratedWeaponDetectionSystem()
        system.run()
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ System shutdown complete")


if __name__ == "__main__":
    main()
