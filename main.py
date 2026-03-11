"""
Real-Time Weapon Detection System with Firebase Realtime Database and Cloudinary
COMPLETE SOLUTION: Pre-buffering + Continuous buffering for guaranteed video upload
Author: FYP Team
Version: 7.4 - FINAL FIX: per-camera buffers, mobile video now uploads
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
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    print("⚠️ Firebase not installed. Run: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

# Cloudinary imports
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_AVAILABLE = True
except ImportError:
    print("⚠️ Cloudinary not installed. Run: pip install cloudinary")
    CLOUDINARY_AVAILABLE = False

from core.integrated_gun_detection_system import IntegratedGunDetectionSystem


class FirebaseRealtimeDB:
    """Firebase Realtime Database handler with Cloudinary video upload"""
    
    # =====================================================================
    # Different cameras with different locations
    # =====================================================================
    
    # 📍 Wazirabad Camera (Laptop)
    CAMERA_WAZIRABAD = {
        'id': 'CAM_WZD_001',
        'name': 'Wazirabad Security Camera',
        'address': 'Wazirabad, Pakistan',
        'city': 'Wazirabad',
        'lat': 32.440418,
        'lng': 74.120255,
        'type': 'Laptop Camera'
    }
    
    # 📍 Gujranwala Camera (Mobile)
    CAMERA_GUJRANWALA = {
        'id': 'CAM_GRW_001',
        'name': 'Gujranwala Security Camera',
        'address': 'Gujranwala, Pakistan',
        'city': 'Gujranwala',
        'lat': 32.187691,
        'lng': 74.194450,
        'type': 'Mobile Camera'
    }

    def __init__(self):
        self.initialized = False
        self.app = None
        self.database_url = 'https://fypiov-default-rtdb.firebaseio.com/'
        self.alert_count = 0
        
        # Cloudinary video buffer configuration
        self.cloudinary_initialized = False
        # =============== PER‑CAMERA BUFFERS ===============
        self.camera_buffers = {}          # dict: camera_id -> list of frames
        self.buffer_max_frames = 150      # per camera, keep last 5 seconds (30fps)
        self.buffer_lock = threading.Lock()  # Thread safety
        # ==================================================
        self.last_alert_time = 0
        self.video_cooldown = 5  # Seconds between uploads
        self.videos_uploaded = 0
        
        # Store camera info by ID for easy lookup (optional)
        self.cameras_by_id = {
            'CAM_WZD_001': self.CAMERA_WAZIRABAD,
            'CAM_GRW_001': self.CAMERA_GUJRANWALA
        }
        
        self._init()
        self._init_cloudinary()

    def _init(self):
        """Initialize Firebase Realtime Database"""
        print("\n" + "=" * 60)
        print("🔥 FIREBASE REALTIME DB INIT")
        print("=" * 60)

        if not FIREBASE_AVAILABLE:
            print("❌ Firebase SDK not available")
            return

        try:
            # Find config file
            config_file = None
            for path in ["serviceAccountKey.json",
                         "config/firebase_config.json",
                         "firebase_config.json"]:
                if os.path.exists(path):
                    config_file = path
                    print(f"✅ Config found: {path}")
                    break

            if not config_file:
                print("❌ No Firebase config found")
                return

            # Clean existing apps
            app_name = 'realtime_alerts_app'
            if firebase_admin._apps:
                print("🔄 Cleaning existing apps...")
                for app in list(firebase_admin._apps.values()):
                    try:
                        firebase_admin.delete_app(app)
                    except:
                        pass

            # Initialize Firebase
            cred = credentials.Certificate(config_file)
            self.app = firebase_admin.initialize_app(cred, {
                'databaseURL': self.database_url
            }, name=app_name)

            # Create references
            self.alerts_ref = db.reference('alerts', app=self.app)
            self.latest_ref = db.reference('latest_alert', app=self.app)
            self.stats_ref = db.reference('system_stats', app=self.app)
            self.cameras_ref = db.reference('cameras', app=self.app)

            # Save both cameras info
            cameras_info = {
                self.CAMERA_WAZIRABAD['id']: {
                    'camera_id': self.CAMERA_WAZIRABAD['id'],
                    'name': self.CAMERA_WAZIRABAD['name'],
                    'address': self.CAMERA_WAZIRABAD['address'],
                    'city': self.CAMERA_WAZIRABAD['city'],
                    'latitude': self.CAMERA_WAZIRABAD['lat'],
                    'longitude': self.CAMERA_WAZIRABAD['lng'],
                    'type': self.CAMERA_WAZIRABAD['type'],
                    'location': {
                        'lat': self.CAMERA_WAZIRABAD['lat'],
                        'lng': self.CAMERA_WAZIRABAD['lng']
                    },
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
                    'location': {
                        'lat': self.CAMERA_GUJRANWALA['lat'],
                        'lng': self.CAMERA_GUJRANWALA['lng']
                    },
                    'installed_at': '2026-03-07',
                    'status': 'active'
                }
            }
            
            self.cameras_ref.set(cameras_info)
            print(f"✅ Cameras saved to Firebase: {len(cameras_info)} cameras")

            # Test connection
            db.reference('test', app=self.app).set({
                'connected': True,
                'timestamp': int(time.time() * 1000),
                'message': 'Realtime DB connected with dual locations!'
            })

            self.initialized = True
            print("✅✅✅ FIREBASE REALTIME DB READY! ✅✅✅")
            print(f"📡 Database URL: {self.database_url}")
            print("\n📍 CAMERA LOCATIONS:")
            print(f"   1. {self.CAMERA_WAZIRABAD['name']}")
            print(f"      Coordinates: {self.CAMERA_WAZIRABAD['lat']}, {self.CAMERA_WAZIRABAD['lng']}")
            print(f"      Address: {self.CAMERA_WAZIRABAD['address']}")
            print(f"   2. {self.CAMERA_GUJRANWALA['name']}")
            print(f"      Coordinates: {self.CAMERA_GUJRANWALA['lat']}, {self.CAMERA_GUJRANWALA['lng']}")
            print(f"      Address: {self.CAMERA_GUJRANWALA['address']}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Firebase init error: {e}")
            import traceback
            traceback.print_exc()

    def _init_cloudinary(self):
        """Initialize Cloudinary for video uploads"""
        if not CLOUDINARY_AVAILABLE:
            print("❌ Cloudinary not available")
            return
            
        try:
            # Configure Cloudinary with your credentials
            cloudinary.config(
                cloud_name="dsnpjwaly",
                api_key="822554518314666",
                api_secret="5Mx7QjuEoe9so37yJLXM3LbJOL0",
                secure=True
            )
            
            # Test connection
            cloudinary.api.ping()
            self.cloudinary_initialized = True
            
            print("=" * 60)
            print("✅✅✅ CLOUDINARY READY! ✅✅✅")
            print(f"☁️ Cloud Name: {cloudinary.config().cloud_name}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Cloudinary init error: {e}")
            self.cloudinary_initialized = False

    # =============== PER‑CAMERA BUFFER METHODS ===============
    def add_frame_to_buffer(self, frame, camera_id=None):
        """Add frame to the buffer of the specified camera."""
        if frame is None or camera_id is None:
            return False
            
        try:
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                return False
            
            with self.buffer_lock:
                # Create buffer for this camera if not exists
                if camera_id not in self.camera_buffers:
                    self.camera_buffers[camera_id] = []
                
                frame_copy = frame.copy()
                self.camera_buffers[camera_id].append({
                    'frame': frame_copy,
                    'timestamp': time.time()
                })
                
                # Limit buffer size
                if len(self.camera_buffers[camera_id]) > self.buffer_max_frames:
                    self.camera_buffers[camera_id].pop(0)
            
            return True
                
        except Exception as e:
            print(f"❌ Error adding frame to buffer: {e}")
            return False

    def get_buffer_size(self, camera_id=None):
        """Get current buffer size for a camera (or total if no camera_id)."""
        with self.buffer_lock:
            if camera_id:
                return len(self.camera_buffers.get(camera_id, []))
            else:
                return sum(len(buf) for buf in self.camera_buffers.values())
    # ==========================================================

    def create_video_from_buffer(self, alert_id, camera_id):
        """Create video from the buffer of a specific camera and upload to Cloudinary."""
        if camera_id is None:
            print("❌ create_video_from_buffer: camera_id is required")
            return None
        
        with self.buffer_lock:
            if camera_id not in self.camera_buffers:
                print(f"❌ No buffer for camera {camera_id}")
                return None
            frames_list = self.camera_buffers[camera_id]
            buffer_size = len(frames_list)
            if buffer_size < 30:
                print(f"⚠️ Not enough frames for camera {camera_id}: {buffer_size} < 30")
                return None
            buffer_copy = [item['frame'].copy() for item in frames_list]
        
        print(f"\n{'🎥' * 20}")
        print(f"🎥 CREATING VIDEO FROM {len(buffer_copy)} FRAMES (Camera: {camera_id})")
        print(f"{'🎥' * 20}\n")
        
        if not self.cloudinary_initialized:
            print("⚠️ Cloudinary not initialized")
            return None
            
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmpfile:
                temp_video_path = tmpfile.name
            
            height, width = buffer_copy[0].shape[:2]
            
            fps = 15.0  # SLOW MOTION
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ VideoWriter failed to open")
                return None
            
            frames_written = 0
            for i, frame in enumerate(buffer_copy):
                if i % 2 == 0:
                    out.write(frame)
                    frames_written += 1
                    if i % 3 == 0:
                        out.write(frame)
                        frames_written += 1
            
            out.release()
            
            original_duration = buffer_size / 30.0
            video_duration = frames_written / fps
            slow_factor = video_duration / original_duration
            
            if not os.path.exists(temp_video_path):
                return None
                
            file_size = os.path.getsize(temp_video_path)
            if file_size == 0:
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            camera_tag = f"cam_{camera_id}" if camera_id else "unknown"
            public_id = f"alert_{self.alert_count}_{timestamp}_{camera_tag}"
            
            upload_result = cloudinary.uploader.upload(
                temp_video_path,
                resource_type="video",
                folder="iov_alerts",
                public_id=public_id,
                tags=["iov", f"alert_{alert_id}", "weapon_detection", camera_tag],
                eager=[{"width": 640, "height": 360, "crop": "scale"}],
                eager_async=False,
                overwrite=True,
                timeout=60
            )
            
            thumbnail_url = None
            if 'eager' in upload_result and len(upload_result['eager']) > 0:
                thumbnail_url = upload_result['eager'][0]['secure_url']
            
            self.videos_uploaded += 1
            
            return {
                'video_url': upload_result['secure_url'],
                'thumbnail_url': thumbnail_url,
                'public_id': upload_result['public_id'],
                'duration': upload_result.get('duration', 0),
                'format': upload_result.get('format', 'mp4'),
                'bytes': upload_result.get('bytes', 0),
                'width': upload_result.get('width', width),
                'height': upload_result.get('height', height),
                'fps': fps,
                'slow_factor': slow_factor,
                'speed': 'SLOW MOTION'
            }
            
        except Exception as e:
            print(f"\n{'❌' * 20}")
            print(f"❌ VIDEO UPLOAD ERROR: {e}")
            print(f"{'❌' * 20}\n")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass

    # ==================== send_alert with camera_info ====================
    def send_alert(self, detection_data: dict, frame=None, camera_info=None) -> bool:
        """Send alert to Firebase Realtime Database with optional video.
           camera_info must be the full camera dictionary (CAMERA_WAZIRABAD or CAMERA_GUJRANWALA)."""
        if not self.initialized:
            print("❌ Firebase not initialized")
            return False

        # Validate camera_info
        if camera_info is None:
            print("❌ CRITICAL ERROR: camera_info is None! Cannot send alert.")
            return False

        try:
            timestamp_ms = int(time.time() * 1000)
            alert_id = f"alert_{self.alert_count}_{timestamp_ms}"
            alert_number = self.alert_count

            weapon_type = detection_data.get('class', 'WEAPON')
            confidence = detection_data.get('confidence', 0.0)
            bbox = detection_data.get('bbox', None)
            
            # =============== USE THE PROVIDED camera_info DIRECTLY ===============
            camera_id = camera_info['id']
            camera_name = camera_info['name']
            camera_address = camera_info['address']
            camera_city = camera_info.get('city', camera_address.split(',')[0])
            camera_lat = camera_info['lat']
            camera_lng = camera_info['lng']
            camera_type = camera_info['type']
            
            # Debug print to verify camera info
            print(f"📹 DEBUG - Sending alert from camera: {camera_name} at {camera_address}")
            print(f"📹 DEBUG - Camera ID: {camera_id}, City: {camera_city}")
            
            # Add frame to per‑camera buffer
            if frame is not None:
                self.add_frame_to_buffer(frame, camera_id)
            
            alert = {
                'id': alert_id,
                'type': str(weapon_type).upper(),
                'weapon_class': str(weapon_type),
                'confidence': round(float(confidence), 4),
                'timestamp': timestamp_ms,
                'time': datetime.now().strftime('%H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'datetime': datetime.now().isoformat(),
                'status': 'active',
                'priority': detection_data.get('threat_level', 'HIGH'),
                'camera_id': camera_id,
                'camera_name': camera_name,
                'camera_type': camera_type,
                'Area': camera_address,
                'city': camera_city,
                'location': {
                    'lat': camera_lat,
                    'lng': camera_lng,
                },
                'location_name': camera_address.split(',')[0],
                'full_address': camera_address,
                'frame_count': detection_data.get('frame_count', 0),
                'description': f"{weapon_type} detected at {camera_address}",
                'acknowledged': False
            }

            # =============== CAMERA‑SPECIFIC BUFFER SIZE ===============
            buffer_size = self.get_buffer_size(camera_id)
            current_time = time.time()
            
            # Create video from buffer with camera-specific frames
            if self.cloudinary_initialized and buffer_size >= 30:
                print(f"✅ Starting video upload for {camera_name}...")
                video_data = self.create_video_from_buffer(alert_number, camera_id)
                
                if video_data and video_data.get('video_url'):
                    alert['video'] = {
                        'url': video_data['video_url'],
                        'thumbnail': video_data.get('thumbnail_url'),
                        'public_id': video_data['public_id'],
                        'duration': video_data.get('duration', 0),
                        'format': video_data.get('format', 'mp4'),
                        'size': video_data.get('bytes', 0),
                        'width': video_data.get('width', 0),
                        'height': video_data.get('height', 0),
                        'fps': video_data.get('fps', 15),
                        'slow_factor': video_data.get('slow_factor', 2.0),
                        'speed': 'SLOW MOTION',
                        'uploaded_at': datetime.now().isoformat()
                    }
                    
                    alert['video_url'] = video_data['video_url']
                    if video_data.get('thumbnail_url'):
                        alert['video_thumbnail'] = video_data['thumbnail_url']
                    alert['video_public_id'] = video_data['public_id']
                    alert['video_fps'] = 15
                    alert['video_speed'] = 'SLOW MOTION'
                    alert['has_video'] = True
                    
                    # Clear old frames from this camera (keep last 30)
                    with self.buffer_lock:
                        if camera_id in self.camera_buffers:
                            # Keep only frames newer than 2 seconds
                            recent = [item for item in self.camera_buffers[camera_id] 
                                      if time.time() - item['timestamp'] < 2.0]
                            self.camera_buffers[camera_id] = recent[-30:]  # keep max 30 most recent
                    
                    self.last_alert_time = current_time
                else:
                    alert['has_video'] = False
            else:
                alert['has_video'] = False

            if bbox and len(bbox) >= 4:
                alert['bbox'] = {
                    'x': int(bbox[0]),
                    'y': int(bbox[1]),
                    'width': int(bbox[2]),
                    'height': int(bbox[3])
                }

            # =============== SEND TO FIREBASE ===============
            self.alerts_ref.child(alert_id).set(alert)
            self.latest_ref.set(alert)
            # =================================================

            self.alert_count += 1
            self.stats_ref.update({
                'last_alert_time': timestamp_ms,
                'last_weapon': str(weapon_type).upper(),
                'last_confidence': round(float(confidence), 4),
                'last_camera': camera_name,
                'last_camera_id': camera_id,
                'last_city': camera_city,
                'last_location': {
                    'lat': camera_lat,
                    'lng': camera_lng,
                    'name': camera_address
                },
                'system_status': 'EMERGENCY',
                'total_alerts': self.alert_count,
                'videos_uploaded': self.videos_uploaded,
                'cloudinary_enabled': self.cloudinary_initialized,
                'video_speed': 'SLOW MOTION (15 FPS)',
                'updated_at': datetime.now().isoformat()
            })

            print(f"\n{'🔥' * 20}")
            print(f"🔥 ALERT SENT TO FIREBASE!")
            print(f"🔥 Alert #{alert_number}")
            print(f"🔥 Type: {weapon_type} | Confidence: {confidence:.2f}")
            print(f"📹 Camera: {camera_name}")
            print(f"📍 Location: {camera_address}")
            print(f"📍 City: {camera_city}")
            if 'video_url' in alert:
                print(f"📹 VIDEO INCLUDED: ✅")
            else:
                print(f"📹 No video (buffer: {buffer_size} frames)")
            print(f"{'🔥' * 20}\n")

            return True

        except Exception as e:
            print(f"❌ Error sending alert: {e}")
            import traceback
            traceback.print_exc()
            return False


# =====================================================================
# Camera Handler with Multi-threading
# =====================================================================
class CameraHandler:
    """Class to manage multiple cameras"""
    
    def __init__(self):
        self.cameras = []
        self.active_cameras = 0
        self.frame_queues = {}
        self.threads = []
        self.running = False
        self.working_cameras = []  # Store only cameras that are working
        
    def add_camera(self, source, camera_info, fps_limit=30):
        """Add new camera - only if it works"""
        try:
            print(f"📹 Testing camera: {camera_info['name']}...")
            
            # Try to open camera
            if isinstance(source, int) or (isinstance(source, str) and source.startswith('http')):
                cap = cv2.VideoCapture(source)
                
                if isinstance(source, str) and 'http' in source:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Lower resolution for mobile for stability
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                
                # Test if camera actually works by reading a frame
                if cap.isOpened():
                    # Try to read a test frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        # Camera is working!
                        camera_data = {
                            'source': source,
                            'cap': cap,
                            'info': camera_info,                 # store full info
                            'name': camera_info['name'],
                            'camera_id': camera_info['id'],
                            'city': camera_info['city'],
                            'fps_limit': fps_limit,
                            'last_frame': None,
                            'frame_count': 0,
                            'fps': 0,
                            'last_time': time.time()
                        }
                        self.cameras.append(camera_data)
                        self.working_cameras.append(camera_data)  # Add to working list
                        self.frame_queues[camera_info['id']] = queue.Queue(maxsize=2)
                        self.active_cameras += 1
                        print(f"✅✅✅ Camera WORKING: {camera_info['name']} at {camera_info['address']}")
                        return True
                    else:
                        print(f"❌ Camera NOT WORKING: {camera_info['name']} - Cannot read frames")
                        cap.release()
                        return False
                else:
                    print(f"❌ Camera NOT WORKING: {camera_info['name']} - Cannot open")
                    return False
        except Exception as e:
            print(f"❌ Error testing camera {camera_info['name']}: {e}")
            return False
    
    def camera_reader_thread(self, camera):
        """Separate thread for each camera"""
        name = camera['name']
        camera_id = camera['camera_id']
        cap = camera['cap']
        
        while self.running:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    if frame.shape[1] > 640:
                        scale = 640 / frame.shape[1]
                        new_width = 640
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    if self.frame_queues[camera_id].full():
                        try:
                            self.frame_queues[camera_id].get_nowait()
                        except:
                            pass
                    
                    # Store frame with camera_id and full camera info
                    self.frame_queues[camera_id].put({
                        'frame': frame,
                        'camera_id': camera_id,
                        'camera_info': camera['info'],   # <-- ADDED full info
                        'timestamp': time.time()
                    })
                    
                    camera['frame_count'] += 1
                    current_time = time.time()
                    if current_time - camera['last_time'] >= 1.0:
                        camera['fps'] = camera['frame_count']
                        camera['frame_count'] = 0
                        camera['last_time'] = current_time
                    
                    time.sleep(1.0 / camera['fps_limit'])
                else:
                    print(f"⚠️ Camera {name} lost connection, trying to reconnect...")
                    # Try to reconnect
                    cap.release()
                    time.sleep(2)
                    cap.open(camera['source'])
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Error in {name} thread: {e}")
                time.sleep(0.1)
    
    def start_all(self):
        """Start all camera threads"""
        self.running = True
        for camera in self.cameras:
            thread = threading.Thread(
                target=self.camera_reader_thread,
                args=(camera,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            print(f"🚀 Started thread for {camera['name']} - {camera['city']}")
    
    def get_frames(self):
        """Get latest frames from all cameras (now includes camera_info)"""
        frames = []
        for camera in self.cameras:
            camera_id = camera['camera_id']
            try:
                if camera_id in self.frame_queues:
                    frame_data = self.frame_queues[camera_id].get_nowait()
                    frame_data['fps'] = camera['fps']
                    # camera_info is already inside frame_data (added in reader thread)
                    frames.append(frame_data)
            except queue.Empty:
                pass
        return frames
    
    def release_all(self):
        """Release all cameras"""
        self.running = False
        time.sleep(0.5)
        
        for camera in self.cameras:
            try:
                camera['cap'].release()
                print(f"✅ Released camera: {camera['name']}")
            except:
                pass


class WeaponDetectionApp:
    """Main application with pre-buffering and continuous frame capture"""
    
    def __init__(self):
        self.firebase_rt = FirebaseRealtimeDB()
        self.last_callback_time = 0
        self.callback_cooldown = 10
        self.total_alerts_sent = 0
        self.frame_count = 0
        self.frames_added = 0
        self.detection_system = None
        
        self.detected_objects = {}
        self.object_timeout = 30
        self.object_colors = {}
        self.total_unique_objects = 0
        
        self.camera_handler = CameraHandler()
        self.last_print_time = 0
        
        # Store last alert time per camera
        self.last_alert_per_camera = {}
        
        # =============== IGNORE LIST - Only weapons, ignore everything else ===============
        self.ignore_classes = ['fire', 'smoke', 'person', 'grenade', 'explosion', 'explosive', 'bomb']

    def get_object_color(self, object_id):
        """Generate unique color for each object"""
        if object_id not in self.object_colors:
            import hashlib
            hash_obj = hashlib.md5(str(object_id).encode())
            hash_hex = hash_obj.hexdigest()
            
            r = int(hash_hex[0:2], 16) % 200 + 55
            g = int(hash_hex[2:4], 16) % 200 + 55
            b = int(hash_hex[4:6], 16) % 200 + 55
            
            self.object_colors[object_id] = (b, g, r)
        
        return self.object_colors[object_id]

    def cleanup_old_objects(self):
        """Remove old detected objects from memory"""
        current_time = time.time()
        expired_ids = []
        
        for obj_id, last_seen in self.detected_objects.items():
            if current_time - last_seen > self.object_timeout:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            del self.detected_objects[obj_id]
            if obj_id in self.object_colors:
                del self.object_colors[obj_id]
            
        if expired_ids and time.time() - self.last_print_time > 5:
            print(f"🧹 Cleaned up {len(expired_ids)} expired objects")
            self.last_print_time = time.time()

    def is_duplicate_detection(self, detection_data, camera_id):
        """Check if this is a duplicate detection for a specific camera"""
        bbox = detection_data.get('bbox', None)
        if bbox is None or len(bbox) < 4:
            return False, None
        
        current_time = time.time()
        
        self.cleanup_old_objects()
        
        # Create a camera-specific object ID
        for obj_id, last_seen in self.detected_objects.items():
            if obj_id.startswith(f"{camera_id}_") and abs(current_time - last_seen) < self.callback_cooldown:
                return True, obj_id
        
        new_obj_id = f"{camera_id}_obj_{self.total_unique_objects}_{int(current_time)}"
        self.detected_objects[new_obj_id] = current_time
        self.total_unique_objects += 1
        
        return False, new_obj_id

    # ==================== MODIFIED: now expects camera_info inside detection_data ====================
    def on_detection_callback(self, detection_data: dict):
        """
        Callback function triggered when weapon is detected
        Args:
            detection_data: Dictionary containing detection information
        """
        current_time = time.time()

        # =============== GET CAMERA INFO FROM DETECTION DATA ===============
        camera_info = detection_data.get('camera_info')
        if camera_info is None:
            print("❌ CRITICAL ERROR: No camera_info in detection_data!")
            return
        
        camera_id = camera_info['id']
        # ====================================================================

        # Debug print to verify which camera detected
        print(f"📱 DEBUG - Detection callback from camera: {camera_info['name']} (ID: {camera_id})")

        # Per-camera cooldown check
        last_time = self.last_alert_per_camera.get(camera_id, 0)
        if current_time - last_time < self.callback_cooldown:
            print(f"⏸️ Cooldown for camera {camera_id}: {current_time - last_time:.1f}s < {self.callback_cooldown}s")
            return

        # Duplicate detection check with camera ID
        is_duplicate, object_id = self.is_duplicate_detection(detection_data, camera_id)
        
        if is_duplicate:
            print(f"⏸️ Duplicate detection ignored for camera {camera_id} (object: {object_id})")
            return

        # Get detection details
        det_class = detection_data.get('class', 'UNKNOWN').lower()
        
        # =============== IGNORE NON-WEAPON ITEMS ===============
        # Check if detected class is in ignore list
        if any(ignore_class in det_class for ignore_class in self.ignore_classes):
            print(f"⏸️ Ignoring {det_class} detection (only weapons allowed)")
            return
        
        # Also check for common weapon keywords
        weapon_keywords = ['gun', 'pistol', 'rifle', 'shotgun', 'weapon', 'knife', 'blade']
        is_weapon = any(weapon in det_class for weapon in weapon_keywords)
        
        if not is_weapon and det_class not in weapon_keywords:
            print(f"⏸️ Ignoring {det_class} (not a weapon)")
            return
        # =======================================================

        # Confidence threshold check
        confidence = detection_data.get('confidence', 0.0)
        if confidence < 0.3:
            print(f"⏸️ Low confidence: {confidence:.2f} < 0.3")
            return
        
        # Add object ID
        if object_id:
            detection_data['object_id'] = object_id
        
        # Add additional info
        detection_data['camera_id'] = camera_id
        detection_data['threat_level'] = 'HIGH' if confidence > 0.7 else 'MEDIUM'
        detection_data['frame_count'] = self.frame_count

        print(f"\n{'🚨' * 20}")
        print(f"🚨 WEAPON DETECTED: {det_class} ({confidence:.2f})")
        print(f"📹 Camera: {camera_info['name']} ({camera_id})")
        if object_id:
            print(f"🆔 Object ID: {object_id}")
        print(f"{'🚨' * 20}\n")

        # Get frame from detection data
        frame = detection_data.get('frame', None)
        
        # Send alert to Firebase with full camera_info
        if self.firebase_rt.send_alert(detection_data, frame, camera_info):
            self.last_alert_per_camera[camera_id] = current_time
            self.last_callback_time = current_time
            self.total_alerts_sent += 1
            print(f"✅ Alert #{self.total_alerts_sent} sent successfully from {camera_info['name']}!")
        else:
            print(f"❌ Failed to send alert from {camera_info['name']}")

    def draw_detections_with_colors(self, frame, detections, camera_id, fps=0):
        """Draw colored boxes around detected objects"""
        if frame is None or detections is None:
            return frame
        
        display_frame = frame.copy()
        
        # Get camera name from ID
        if camera_id == 'CAM_WZD_001':
            camera_name = "Wazirabad"
            color = (0, 255, 255)  # Yellow
        elif camera_id == 'CAM_GRW_001':
            camera_name = "Gujranwala"
            color = (255, 0, 255)  # Purple
        else:
            camera_name = "Unknown"
            color = (255, 255, 255)  # White
        
        location_text = f"{camera_name} Camera | FPS: {fps}"
        cv2.putText(
            display_frame,
            location_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        for detection in detections:
            bbox = detection.get('bbox')
            if bbox is None or len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox[:4])
            confidence = detection.get('confidence', 0)
            class_name = detection.get('class', 'Unknown').lower()
            
            # =============== Only draw boxes for weapons ===============
            # Check if it's a weapon (not in ignore list)
            is_weapon = True
            for ignore_class in self.ignore_classes:
                if ignore_class in class_name:
                    is_weapon = False
                    break
            
            if is_weapon:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                label = f"{class_name} {confidence:.2f}"
                
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    display_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width + 10, y1),
                    (0, 0, 255),
                    -1
                )
                
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            # ============================================================
        
        return display_frame

    def combine_frames(self, frames_list):
        """Combine camera feeds into one screen"""
        if not frames_list:
            return None
        
        processed_frames = []
        for frame_data in frames_list:
            if frame_data and frame_data['frame'] is not None:
                frame = cv2.resize(frame_data['frame'], (640, 480))
                processed_frames.append({
                    'frame': frame,
                    'camera_id': frame_data['camera_id'],
                    'fps': frame_data['fps']
                })
        
        # Handle different numbers of cameras
        if len(processed_frames) == 2:
            combined = np.hstack([
                processed_frames[0]['frame'],
                processed_frames[1]['frame']
            ])
            
            # Add status text
            cv2.putText(
                combined,
                f"Total Alerts: {self.total_alerts_sent} | Videos: {self.firebase_rt.videos_uploaded}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add camera labels with colors
            cv2.putText(
                combined,
                "📍 WAZIRABAD (Laptop)",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),  # Yellow
                2
            )
            cv2.putText(
                combined,
                "📍 GUJRANWALA (Mobile)",
                (650, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),  # Purple
                2
            )
            
            # Add FPS for each camera
            if len(processed_frames) >= 2:
                cv2.putText(
                    combined,
                    f"FPS: {processed_frames[0]['fps']}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    combined,
                    f"FPS: {processed_frames[1]['fps']}",
                    (650, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            return combined
        elif len(processed_frames) == 1:
            # Single camera view
            frame = processed_frames[0]['frame']
            cv2.putText(
                frame,
                f"Alerts: {self.total_alerts_sent} | Videos: {self.firebase_rt.videos_uploaded}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            return frame
        else:
            return None

    def setup_cameras(self):
        """Setup cameras with locations - only add working cameras"""
        print("\n" + "=" * 60)
        print("📹 SETTING UP CAMERAS WITH LOCATIONS")
        print("=" * 60)
        print("🔍 Testing each camera... Only working cameras will be added")
        print("-" * 60)
        
        working_count = 0
        
        # 1. Try Wazirabad Camera (Laptop)
        if self.camera_handler.add_camera(0, self.firebase_rt.CAMERA_WAZIRABAD, fps_limit=30):
            working_count += 1
            print(f"  ✅ {self.firebase_rt.CAMERA_WAZIRABAD['name']} WORKING (30 FPS)")
            print(f"     📍 City: {self.firebase_rt.CAMERA_WAZIRABAD['city']}")
        else:
            print(f"  ⚠️ {self.firebase_rt.CAMERA_WAZIRABAD['name']} NOT WORKING - Skipping")
        
        # 2. Try Gujranwala Camera (Mobile IP Webcam) - Optimized for stability
        mobile_url = "http://192.168.1.4:8080/video?fps=15&resolution=320x240"
        if self.camera_handler.add_camera(mobile_url, self.firebase_rt.CAMERA_GUJRANWALA, fps_limit=15):
            working_count += 1
            print(f"  ✅ {self.firebase_rt.CAMERA_GUJRANWALA['name']} WORKING (15 FPS)")
            print(f"     📍 City: {self.firebase_rt.CAMERA_GUJRANWALA['city']}")
        else:
            print(f"  ⚠️ {self.firebase_rt.CAMERA_GUJRANWALA['name']} NOT WORKING - Skipping")
            print("     📱 Make sure IP Webcam is running on your phone and connected to same network")
        
        print("-" * 60)
        print(f"\n📊 WORKING CAMERAS: {working_count} / 2")
        
        if working_count == 0:
            print("❌ No working cameras found!")
            return False
        
        print("✅ Starting camera threads for working cameras...")
        self.camera_handler.start_all()
        time.sleep(1)  # Give cameras time to start
        
        return True

    def run(self):
        """Main application loop"""
        
        print("\n" + "=" * 60)
        print("🎯 WEAPON DETECTION SYSTEM - DUAL LOCATION (FIXED)")
        print("=" * 60)
        print("🔧 FIXES APPLIED:")
        print("   ✅ Only weapons detected (ignoring fire/smoke/person/grenade)")
        print("   ✅ Only working cameras used")
        print("   ✅ Proper location mapping (Wazirabad/Gujranwala) - now using explicit camera_info")
        print("   ✅ Detection data extraction fixed (class from meta, confidence from weapon fields)")
        print("   ✅ Removed automatic callback registration to avoid wrong location alerts")
        print("   ✅ Per‑camera buffers – mobile video now uploads correctly")
        print("=" * 60)
        print(f"🔥 Firebase: {'✅' if self.firebase_rt.initialized else '❌'}")
        print(f"☁️ Cloudinary: {'✅' if self.firebase_rt.cloudinary_initialized else '❌'}")
        print("=" * 60)

        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        try:
            print("🚀 Initializing detection system...")
            self.detection_system = IntegratedGunDetectionSystem(model_path)
            # ❌ REMOVED: self.detection_system.add_detection_callback(self.on_detection_callback)
            # We call on_detection_callback manually with correct camera_info.
            print("✅ Detection system initialized (no automatic callback)\n")

            if not self.setup_cameras():
                print("❌ No cameras available")
                return

            if self.firebase_rt.cloudinary_initialized:
                print(f"\n{'📹' * 20}")
                print("📹 PRE-BUFFERING FRAMES...")
                print(f"{'📹' * 20}\n")
                
                prebuffer_count = 0
                prebuffer_target = 60
                
                while prebuffer_count < prebuffer_target:
                    frames = self.camera_handler.get_frames()
                    for frame_data in frames:
                        if frame_data['frame'] is not None:
                            self.firebase_rt.add_frame_to_buffer(frame_data['frame'], frame_data['camera_id'])
                            prebuffer_count += 1
                    
                    if prebuffer_count < prebuffer_target:
                        time.sleep(0.01)
                
                # Print per‑camera buffer sizes after pre‑buffering
                for cam in self.camera_handler.working_cameras:
                    cam_id = cam['camera_id']
                    buf_size = self.firebase_rt.get_buffer_size(cam_id)
                    print(f"   📊 {cam['name']} buffer: {buf_size} frames")
                print(f"✅ PRE-BUFFER COMPLETE!\n")

            print("=" * 60)
            print("🚀 STARTING WEAPON DETECTION...")
            print("=" * 60)
            print("📍 WORKING CAMERAS:")
            for cam in self.camera_handler.working_cameras:
                print(f"   ✅ {cam['name']} - {cam['city']}")
            print("=" * 60)
            print("⚠️ IGNORING: Fire, Smoke, Person, Grenade, Explosion")
            print("🎯 ONLY DETECTING: Weapons (Guns, Knives, etc.)")
            print("=" * 60)
            print("Press 'q' to quit\n")
            
            last_frame_time = time.time()
            frame_count = 0
            display_fps = 0
            
            while True:
                loop_start = time.time()
                
                frames_data = self.camera_handler.get_frames()
                
                processed_frames = []
                for frame_data in frames_data:
                    frame = frame_data['frame']
                    camera_id = frame_data['camera_id']
                    camera_info = frame_data['camera_info']   # we have full info
                    camera_fps = frame_data['fps']
                    
                    if frame is None:
                        continue
                    
                    self.frame_count += 1
                    
                    if self.firebase_rt.cloudinary_initialized:
                        self.firebase_rt.add_frame_to_buffer(frame, camera_id)
                        self.frames_added += 1
                    
                    try:
                        # Detect objects
                        detections, fire_smoke = self.detection_system.detect_objects(frame)
                        
                        # Process detections (optional, but needed for internal logic)
                        results = self.detection_system.process_detections(detections, frame)
                        
                        # =============== FIXED: EXTRACT DETECTION DATA CORRECTLY ===============
                        weapon_detections = []
                        for detection in detections:
                            # Get class name from meta
                            meta = detection.get('meta', {})
                            det_class = meta.get('class_name', '').lower()
                            
                            # Get confidence – try weapon-specific fields first
                            confidence = (
                                detection.get('gun_conf', 0) or
                                detection.get('knife_conf', 0) or
                                detection.get('explosion_conf', 0) or
                                detection.get('grenade_conf', 0) or
                                meta.get('raw_confidence', 0)
                            )
                            
                            # Skip if not a weapon
                            if any(ignore in det_class for ignore in self.ignore_classes):
                                continue
                            weapon_keywords = ['gun', 'pistol', 'rifle', 'shotgun', 'weapon', 'knife', 'blade']
                            if not any(kw in det_class for kw in weapon_keywords):
                                continue
                            
                            # Confidence threshold
                            if confidence < 0.3:
                                continue
                            
                            weapon_detections.append(detection)
                            
                            # Create alert data with correct camera_info
                            alert_data = {
                                'class': det_class.upper(),
                                'confidence': confidence,
                                'bbox': detection.get('bbox', None),
                                'type': 'WEAPON',
                                'camera_id': camera_id,
                                'camera_info': camera_info,
                                'frame': frame,
                                'timestamp': time.time()
                            }
                            
                            # Call callback for weapons only
                            self.on_detection_callback(alert_data)
                        # ========================================================================
                        
                        # Generate system alerts (optional)
                        self.detection_system.generate_detection_alerts(detections, fire_smoke, results)
                        
                        # Draw detections on frame (weapons only)
                        display_frame = self.draw_detections_with_colors(
                            frame, weapon_detections, camera_id, camera_fps
                        )
                        
                        processed_frames.append({
                            'frame': display_frame,
                            'camera_id': camera_id,
                            'fps': camera_fps
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Error on camera {camera_id}: {e}")
                
                # Combine frames for display
                combined_frame = self.combine_frames(processed_frames)
                
                if combined_frame is not None:
                    frame_count += 1
                    if time.time() - last_frame_time >= 1.0:
                        display_fps = frame_count
                        frame_count = 0
                        last_frame_time = time.time()
                    
                    # Add status info
                    working_text = f"Working Cameras: {len(self.camera_handler.working_cameras)}"
                    ignore_text = "Ignoring: Fire/Smoke/Person/Grenade"
                    
                    cv2.putText(
                        combined_frame,
                        working_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.putText(
                        combined_frame,
                        ignore_text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                    
                    cv2.imshow("Weapon Detection - Working Cameras Only - Press 'q' to quit", combined_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Quit signal received")
                    break
                
                loop_time = time.time() - loop_start
                if loop_time < 0.03:
                    time.sleep(0.03 - loop_time)
            
            self.camera_handler.release_all()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\n👋 Stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ System error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "=" * 60)
            print("📊 FINAL STATISTICS")
            print("=" * 60)
            print(f"   Total alerts sent: {self.total_alerts_sent}")
            print(f"   Videos uploaded: {self.firebase_rt.videos_uploaded}")
            
            # Count alerts per camera
            wazirabad_alerts = 0
            gujranwala_alerts = 0
            
            for cam_id, last_time in self.last_alert_per_camera.items():
                if 'WZD' in cam_id:
                    wazirabad_alerts += 1
                elif 'GRW' in cam_id:
                    gujranwala_alerts += 1
            
            print(f"   Wazirabad alerts: {wazirabad_alerts}")
            print(f"   Gujranwala alerts: {gujranwala_alerts}")
            print(f"   Working cameras: {len(self.camera_handler.working_cameras)}")
            print("=" * 60)

            # Update Firebase status
            if self.firebase_rt.initialized:
                try:
                    self.firebase_rt.stats_ref.update({
                        'system_status': 'OFFLINE',
                        'shutdown_time': datetime.now().isoformat(),
                        'total_alerts': self.total_alerts_sent,
                        'videos_uploaded': self.firebase_rt.videos_uploaded,
                        'active_cameras': len(self.camera_handler.working_cameras),
                        'wazirabad_alerts': wazirabad_alerts,
                        'gujranwala_alerts': gujranwala_alerts
                    })
                    print("✅ Firebase status updated")
                except Exception as e:
                    print(f"⚠️ Could not update Firebase: {e}")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("🎯 WEAPON DETECTION SYSTEM - FINAL FIXED VERSION (v7.4)")
    print("=" * 70)
    print("📍 Camera 1: Wazirabad (CAM_WZD_001) - Laptop Camera")
    print("📍 Camera 2: Gujranwala (CAM_GRW_001) - Mobile IP Webcam")
    print("-" * 70)
    print("🔧 FIXES:")
    print("   ✅ Only weapons trigger alerts")
    print("   ✅ Fire/Smoke/Person/Grenade/Explosion ignored")
    print("   ✅ Only working cameras used")
    print("   ✅ Proper location mapping (camera_info explicitly passed)")
    print("   ✅ Detection data correctly extracted from integrated system")
    print("   ✅ Automatic callback removed – no more wrong-location alerts")
    print("   ✅ Per‑camera buffers – video always from correct camera")
    print("=" * 70 + "\n")
    
    app = WeaponDetectionApp()
    app.run()


if __name__ == "__main__":
    main()