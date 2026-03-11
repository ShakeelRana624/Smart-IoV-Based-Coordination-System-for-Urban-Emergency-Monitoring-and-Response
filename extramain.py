"""
Real-Time Weapon Detection System with Firebase Realtime Database and Cloudinary
COMPLETE FIXED VERSION: Camera Location Issue Fixed
Author: FYP Team
Version: 9.0 - CAMERA LOCATION FIXED (Wazirabad + Gujranwala)
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

# Detection System Import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False


class WeaponDetector:
    """سادہ ویپن ڈیٹیکٹر - صرف YOLO"""
    
    def __init__(self, model_path):
        if not YOLO_AVAILABLE:
            raise Exception("YOLO not available!")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.25
        
        # کلاس نام جو ماڈل میں ہیں
        self.class_names = self.model.names
        print(f"✅ Model loaded: {model_path}")
        print(f"📋 Classes: {self.class_names}")
        
    def detect(self, frame):
        """ڈیٹیکشن کریں"""
        if frame is None:
            return []
        
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id
                    })
        
        return detections


class FirebaseRealtimeDB:
    """Firebase Realtime Database handler"""
    
    # =====================================================================
    # 📍 کیمرہ لوکیشنز - وزیرآباد اور گوجرانوالہ
    # =====================================================================
    
    CAMERA_WAZIRABAD = {
        'id': 'CAM_WZD_001',
        'name': 'Wazirabad Security Camera',
        'address': 'Wazirabad, Pakistan',
        'lat': 32.440418,
        'lng': 74.120255,
        'type': 'Laptop Camera'
    }
    
    CAMERA_GUJRANWALA = {
        'id': 'CAM_GRW_001',
        'name': 'Gujranwala Security Camera',
        'address': 'Gujranwala, Pakistan',
        'lat': 32.187691,
        'lng': 74.194450,
        'type': 'Mobile Camera'
    }

    # نظرانداز کلاسز
    IGNORED_CLASSES = ['grenade', 'explosion', 'fire', 'smoke', 'person', 'bomb', 'flame', 'blast']

    def __init__(self):
        self.initialized = False
        self.app = None
        self.database_url = 'https://fypiov-default-rtdb.firebaseio.com/'
        self.alert_count = 0
        
        self.cloudinary_initialized = False
        self.video_buffer = []
        self.buffer_max_frames = 180
        self.buffer_lock = threading.Lock()
        self.videos_uploaded = 0
        
        # ہر کیمرے کا علیحدہ بفر
        self.camera_buffers = {
            'CAM_WZD_001': [],
            'CAM_GRW_001': []
        }
        
        self._init()
        self._init_cloudinary()

    def _init(self):
        """Initialize Firebase"""
        print("\n" + "=" * 60)
        print("🔥 FIREBASE INIT")
        print("=" * 60)

        if not FIREBASE_AVAILABLE:
            print("❌ Firebase not available")
            return

        try:
            config_file = None
            for path in ["serviceAccountKey.json", "config/firebase_config.json", "firebase_config.json"]:
                if os.path.exists(path):
                    config_file = path
                    break

            if not config_file:
                print("❌ No Firebase config")
                return

            app_name = 'weapon_detection_app'
            if firebase_admin._apps:
                for app in list(firebase_admin._apps.values()):
                    try:
                        firebase_admin.delete_app(app)
                    except:
                        pass

            cred = credentials.Certificate(config_file)
            self.app = firebase_admin.initialize_app(cred, {
                'databaseURL': self.database_url
            }, name=app_name)

            self.alerts_ref = db.reference('alerts', app=self.app)
            self.latest_ref = db.reference('latest_alert', app=self.app)
            self.stats_ref = db.reference('system_stats', app=self.app)
            self.cameras_ref = db.reference('cameras', app=self.app)

            # کیمرے سیو کریں
            self.cameras_ref.set({
                self.CAMERA_WAZIRABAD['id']: {
                    'camera_id': self.CAMERA_WAZIRABAD['id'],
                    'name': self.CAMERA_WAZIRABAD['name'],
                    'address': self.CAMERA_WAZIRABAD['address'],
                    'latitude': self.CAMERA_WAZIRABAD['lat'],
                    'longitude': self.CAMERA_WAZIRABAD['lng'],
                    'type': self.CAMERA_WAZIRABAD['type'],
                    'location': {'lat': self.CAMERA_WAZIRABAD['lat'], 'lng': self.CAMERA_WAZIRABAD['lng']},
                    'status': 'active'
                },
                self.CAMERA_GUJRANWALA['id']: {
                    'camera_id': self.CAMERA_GUJRANWALA['id'],
                    'name': self.CAMERA_GUJRANWALA['name'],
                    'address': self.CAMERA_GUJRANWALA['address'],
                    'latitude': self.CAMERA_GUJRANWALA['lat'],
                    'longitude': self.CAMERA_GUJRANWALA['lng'],
                    'type': self.CAMERA_GUJRANWALA['type'],
                    'location': {'lat': self.CAMERA_GUJRANWALA['lat'], 'lng': self.CAMERA_GUJRANWALA['lng']},
                    'status': 'active'
                }
            })

            self.initialized = True
            print("✅ FIREBASE READY!")
            print(f"📡 URL: {self.database_url}")

        except Exception as e:
            print(f"❌ Firebase error: {e}")
            import traceback
            traceback.print_exc()

    def _init_cloudinary(self):
        """Initialize Cloudinary"""
        if not CLOUDINARY_AVAILABLE:
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
            print("✅ CLOUDINARY READY!")
        except Exception as e:
            print(f"❌ Cloudinary error: {e}")

    def is_weapon(self, class_name):
        """چیک کریں کہ ویپن ہے یا نہیں"""
        if not class_name:
            return False
        
        class_lower = str(class_name).lower()
        
        # نظرانداز
        for ignored in self.IGNORED_CLASSES:
            if ignored in class_lower:
                return False
        
        # ویپن کیورڈز
        weapon_keywords = ['gun', 'pistol', 'rifle', 'weapon', 'handgun', 'firearm', 
                          'revolver', 'shotgun', 'ak47', 'knife', 'blade', 'sword']
        
        for kw in weapon_keywords:
            if kw in class_lower:
                return True
        
        # اگر person/fire/smoke نہیں تو ویپن سمجھو
        if class_lower not in ['person', 'fire', 'smoke', 'background', 'people']:
            return True
            
        return False

    def add_frame_to_buffer(self, frame, camera_id):
        """کیمرہ سپیسیفک بفر میں فریم ڈالیں"""
        if frame is None or camera_id is None:
            return
            
        try:
            with self.buffer_lock:
                if camera_id not in self.camera_buffers:
                    self.camera_buffers[camera_id] = []
                
                self.camera_buffers[camera_id].append({
                    'frame': frame.copy(),
                    'timestamp': time.time()
                })
                
                # حد
                while len(self.camera_buffers[camera_id]) > 90:  # 3 سیکنڈ
                    self.camera_buffers[camera_id].pop(0)
        except:
            pass

    def get_buffer_size(self, camera_id=None):
        """بفر سائز"""
        with self.buffer_lock:
            if camera_id:
                return len(self.camera_buffers.get(camera_id, []))
            return sum(len(buf) for buf in self.camera_buffers.values())

    def create_video(self, camera_id):
        """مخصوص کیمرے کی ویڈیو بنائیں"""
        with self.buffer_lock:
            if camera_id not in self.camera_buffers:
                return None
            
            frames = self.camera_buffers[camera_id]
            if len(frames) < 10:
                return None
            
            buffer_copy = [f['frame'].copy() for f in frames]
        
        print(f"🎥 Creating video from {len(buffer_copy)} frames (Camera: {camera_id})")
        
        if not self.cloudinary_initialized:
            return None
            
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_path = tmp.name
            
            h, w = buffer_copy[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (w, h))
            
            for frame in buffer_copy:
                out.write(frame)
            out.release()
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            public_id = f"alert_{self.alert_count}_{camera_id}_{timestamp}"
            
            result = cloudinary.uploader.upload(
                temp_path,
                resource_type="video",
                folder="weapon_alerts",
                public_id=public_id,
                timeout=120
            )
            
            self.videos_uploaded += 1
            print(f"✅ Video uploaded: {result['secure_url']}")
            
            return {
                'video_url': result['secure_url'],
                'public_id': result['public_id']
            }
            
        except Exception as e:
            print(f"❌ Video error: {e}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def send_alert(self, detection_data: dict, frame, camera_info: dict) -> bool:
        """
        🚨 الرٹ بھیجیں - کیمرہ انفو REQUIRED
        """
        if not self.initialized:
            print("❌ Firebase not initialized")
            return False

        if camera_info is None:
            print("❌ ERROR: camera_info is None!")
            return False

        try:
            timestamp_ms = int(time.time() * 1000)
            alert_id = f"alert_{self.alert_count}_{timestamp_ms}"
            
            weapon_type = detection_data.get('class', 'WEAPON')
            confidence = detection_data.get('confidence', 0.0)
            bbox = detection_data.get('bbox', None)
            
            # ========================================
            # 🔴 یقینی بنائیں کہ صحیح کیمرہ ہے
            # ========================================
            camera_id = camera_info['id']
            camera_name = camera_info['name']
            camera_address = camera_info['address']
            camera_lat = camera_info['lat']
            camera_lng = camera_info['lng']
            camera_type = camera_info['type']
            
            print(f"\n🔍 ALERT DEBUG:")
            print(f"   Camera ID: {camera_id}")
            print(f"   Camera Name: {camera_name}")
            print(f"   Camera Address: {camera_address}")
            print(f"   Camera Type: {camera_type}")
            
            # الرٹ ڈیٹا
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
                'priority': 'HIGH' if confidence > 0.6 else 'MEDIUM',
                
                # 🔴 کیمرہ انفو - صحیح
                'camera_id': camera_id,
                'camera_name': camera_name,
                'camera_type': camera_type,
                'Area': camera_address,
                
                # 🔴 لوکیشن - صحیح
                'location': {
                    'lat': camera_lat,
                    'lng': camera_lng,
                },
                'location_name': camera_address.split(',')[0],
                'full_address': camera_address,
                
                'description': f"{weapon_type} detected at {camera_address}",
                'acknowledged': False,
                'has_video': False
            }

            # ویڈیو اپلوڈ
            if self.cloudinary_initialized and self.get_buffer_size(camera_id) >= 10:
                video_data = self.create_video(camera_id)
                if video_data:
                    alert['video_url'] = video_data['video_url']
                    alert['video_public_id'] = video_data['public_id']
                    alert['has_video'] = True

            # باکس
            if bbox and len(bbox) >= 4:
                alert['bbox'] = {
                    'x': int(bbox[0]),
                    'y': int(bbox[1]),
                    'width': int(bbox[2] - bbox[0]),
                    'height': int(bbox[3] - bbox[1])
                }

            # Firebase میں بھیجیں
            self.alerts_ref.child(alert_id).set(alert)
            self.latest_ref.set(alert)

            self.alert_count += 1
            
            # Stats
            self.stats_ref.update({
                'last_alert_time': timestamp_ms,
                'last_weapon': str(weapon_type).upper(),
                'last_confidence': round(float(confidence), 4),
                'last_camera': camera_name,
                'last_camera_id': camera_id,
                'last_location': {
                    'lat': camera_lat,
                    'lng': camera_lng,
                    'name': camera_address
                },
                'system_status': 'ACTIVE',
                'total_alerts': self.alert_count,
                'videos_uploaded': self.videos_uploaded
            })

            print(f"\n{'🔥' * 25}")
            print(f"🔥 ALERT #{self.alert_count} SENT!")
            print(f"🔫 Weapon: {weapon_type} ({confidence:.0%})")
            print(f"📹 Camera: {camera_name}")
            print(f"📍 Location: {camera_address}")
            print(f"🗺️ Coordinates: {camera_lat}, {camera_lng}")
            print(f"📹 Video: {'✅' if alert['has_video'] else '❌'}")
            print(f"{'🔥' * 25}\n")

            return True

        except Exception as e:
            print(f"❌ Alert error: {e}")
            import traceback
            traceback.print_exc()
            return False


class CameraHandler:
    """کیمرہ ہینڈلر"""
    
    def __init__(self):
        self.cameras = {}
        self.frame_queues = {}
        self.threads = []
        self.running = False
        
    def add_camera(self, source, camera_info, fps_limit=30):
        """کیمرہ شامل کریں"""
        camera_id = camera_info['id']
        
        try:
            print(f"📷 Adding: {camera_info['name']} ({camera_id})")
            
            cap = cv2.VideoCapture(source)
            
            if isinstance(source, str) and 'http' in source:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    self.cameras[camera_id] = {
                        'source': source,
                        'cap': cap,
                        'info': camera_info,
                        'fps_limit': fps_limit,
                        'fps': 0,
                        'frame_count': 0,
                        'last_time': time.time(),
                        'last_frame': None
                    }
                    self.frame_queues[camera_id] = queue.Queue(maxsize=3)
                    
                    print(f"✅ Camera ready: {camera_info['name']}")
                    print(f"   📍 {camera_info['address']}")
                    return True
            
            print(f"❌ Failed: {camera_info['name']}")
            return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def camera_thread(self, camera_id):
        """کیمرہ تھریڈ"""
        camera = self.cameras[camera_id]
        cap = camera['cap']
        
        while self.running:
            try:
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # ریسائز
                    if frame.shape[1] > 640:
                        scale = 640 / frame.shape[1]
                        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
                    
                    # قطار میں ڈالیں
                    q = self.frame_queues[camera_id]
                    if q.full():
                        try:
                            q.get_nowait()
                        except:
                            pass
                    q.put(frame)
                    camera['last_frame'] = frame
                    
                    # FPS
                    camera['frame_count'] += 1
                    if time.time() - camera['last_time'] >= 1.0:
                        camera['fps'] = camera['frame_count']
                        camera['frame_count'] = 0
                        camera['last_time'] = time.time()
                    
                    time.sleep(1.0 / camera['fps_limit'])
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                time.sleep(0.5)
    
    def start_all(self):
        """سب شروع"""
        self.running = True
        for camera_id in self.cameras:
            t = threading.Thread(target=self.camera_thread, args=(camera_id,), daemon=True)
            t.start()
            self.threads.append(t)
            print(f"🚀 Started: {self.cameras[camera_id]['info']['name']}")
    
    def get_all_frames(self):
        """تمام فریم"""
        frames = []
        for camera_id, camera in self.cameras.items():
            try:
                q = self.frame_queues[camera_id]
                try:
                    frame = q.get_nowait()
                except queue.Empty:
                    frame = camera['last_frame']
                
                if frame is not None:
                    frames.append({
                        'frame': frame,
                        'camera_id': camera_id,
                        'camera_info': camera['info'],
                        'fps': camera['fps']
                    })
            except:
                pass
        return frames
    
    def release_all(self):
        """ریلیز"""
        self.running = False
        time.sleep(0.5)
        for cam in self.cameras.values():
            try:
                cam['cap'].release()
            except:
                pass


class WeaponDetectionApp:
    """مین ایپلیکیشن"""
    
    def __init__(self):
        self.firebase = FirebaseRealtimeDB()
        self.camera_handler = CameraHandler()
        self.detector = None
        
        # ہر کیمرے کا علیحدہ کولڈاؤن
        self.camera_cooldowns = {}
        self.cooldown_seconds = 10
        
        self.total_alerts = 0
        self.frame_count = 0

    def can_send_alert(self, camera_id):
        """کولڈاؤن چیک"""
        now = time.time()
        last = self.camera_cooldowns.get(camera_id, 0)
        return (now - last) >= self.cooldown_seconds

    def update_cooldown(self, camera_id):
        """کولڈاؤن اپڈیٹ"""
        self.camera_cooldowns[camera_id] = time.time()

    def draw_boxes(self, frame, detections, camera_info, fps):
        """باکس بنائیں"""
        if frame is None:
            return frame
        
        display = frame.copy()
        
        # کیمرہ لیبل
        label = f"{camera_info['name']} | {camera_info['address'].split(',')[0]} | FPS: {fps}"
        cv2.rectangle(display, (0, 0), (500, 30), (0, 0, 0), -1)
        cv2.putText(display, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        weapons = 0
        for det in detections:
            if not self.firebase.is_weapon(det.get('class')):
                continue
            
            bbox = det.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            
            conf = det.get('confidence', 0)
            if conf < 0.25:
                continue
            
            x1, y1, x2, y2 = map(int, bbox[:4])
            weapons += 1
            
            # 🔴 سرخ باکس
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # کارنرز
            cl = 20
            cv2.line(display, (x1, y1), (x1+cl, y1), (0, 0, 255), 5)
            cv2.line(display, (x1, y1), (x1, y1+cl), (0, 0, 255), 5)
            cv2.line(display, (x2, y1), (x2-cl, y1), (0, 0, 255), 5)
            cv2.line(display, (x2, y1), (x2, y1+cl), (0, 0, 255), 5)
            cv2.line(display, (x1, y2), (x1+cl, y2), (0, 0, 255), 5)
            cv2.line(display, (x1, y2), (x1, y2-cl), (0, 0, 255), 5)
            cv2.line(display, (x2, y2), (x2-cl, y2), (0, 0, 255), 5)
            cv2.line(display, (x2, y2), (x2, y2-cl), (0, 0, 255), 5)
            
            # لیبل
            txt = f"WEAPON {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1-th-10), (x1+tw+10, y1), (0, 0, 255), -1)
            cv2.putText(display, txt, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # وارننگ
        if weapons > 0:
            warn = f"⚠ {weapons} WEAPON(S) DETECTED!"
            ts, _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = (display.shape[1] - ts[0]) // 2
            cv2.rectangle(display, (tx-10, 35), (tx+ts[0]+10, 65), (0, 0, 255), -1)
            cv2.putText(display, warn, (tx, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display

    def combine_frames(self, frames_data):
        """فریمز جوڑیں"""
        if not frames_data:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No cameras...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return blank
        
        processed = []
        for fd in frames_data:
            if fd and fd.get('frame') is not None:
                f = cv2.resize(fd['frame'], (640, 480))
                processed.append({'frame': f, 'camera_info': fd['camera_info']})
        
        if len(processed) == 2:
            combined = np.hstack([processed[0]['frame'], processed[1]['frame']])
            
            # لیبلز
            cv2.putText(combined, "WAZIRABAD (Laptop)", (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(combined, "GUJRANWALA (Mobile)", (650, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
        elif len(processed) == 1:
            combined = processed[0]['frame']
        else:
            combined = np.zeros((480, 1280, 3), dtype=np.uint8)
        
        # سٹیٹس
        status = f"Alerts: {self.total_alerts} | Videos: {self.firebase.videos_uploaded}"
        cv2.rectangle(combined, (0, 0), (300, 25), (0, 0, 0), -1)
        cv2.putText(combined, status, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return combined

    def setup_cameras(self):
        """کیمرے سیٹ اپ"""
        print("\n" + "=" * 60)
        print("📹 CAMERA SETUP")
        print("=" * 60)
        
        # 1. وزیرآباد (لیپ ٹاپ)
        self.camera_handler.add_camera(0, self.firebase.CAMERA_WAZIRABAD, fps_limit=30)
        
        # 2. گوجرانوالہ (موبائل)
        mobile_urls = [
            "http://192.168.1.7:8080/video",
            "http://192.168.1.7:4747/video",
        ]
        
        for url in mobile_urls:
            print(f"  📱 Trying: {url}")
            if self.camera_handler.add_camera(url, self.firebase.CAMERA_GUJRANWALA, fps_limit=15):
                break
        
        print(f"\n📊 Active: {len(self.camera_handler.cameras)} cameras")
        
        if self.camera_handler.cameras:
            self.camera_handler.start_all()
            time.sleep(1)
        
        return len(self.camera_handler.cameras) > 0

    def run(self):
        """مین لوپ"""
        
        print("\n" + "=" * 60)
        print("🎯 WEAPON DETECTION v9.0 - LOCATION FIXED")
        print("=" * 60)
        print(f"🔥 Firebase: {'✅' if self.firebase.initialized else '❌'}")
        print(f"☁️ Cloudinary: {'✅' if self.firebase.cloudinary_initialized else '❌'}")
        print("=" * 60)

        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        try:
            print("🚀 Loading model...")
            self.detector = WeaponDetector(model_path)
            print("✅ Model ready\n")

            if not self.setup_cameras():
                print("❌ No cameras")
                return

            # پری بفر
            print("📹 Pre-buffering...")
            for _ in range(30):
                frames = self.camera_handler.get_all_frames()
                for fd in frames:
                    self.firebase.add_frame_to_buffer(fd['frame'], fd['camera_id'])
                time.sleep(0.03)
            print("✅ Buffer ready\n")

            print("=" * 60)
            print("🚀 DETECTION STARTED - Press 'q' to quit")
            print("=" * 60 + "\n")
            
            while True:
                frames_data = self.camera_handler.get_all_frames()
                processed_frames = []
                
                for fd in frames_data:
                    frame = fd['frame']
                    camera_id = fd['camera_id']
                    camera_info = fd['camera_info']
                    fps = fd['fps']
                    
                    if frame is None:
                        continue
                    
                    self.frame_count += 1
                    
                    # بفر میں ڈالیں
                    self.firebase.add_frame_to_buffer(frame, camera_id)
                    
                    try:
                        # ڈیٹیکشن
                        detections = self.detector.detect(frame)
                        
                        # ویپنز فلٹر کریں
                        weapons = [d for d in detections if self.firebase.is_weapon(d.get('class'))]
                        
                        # ========================================
                        # 🔴 الرٹ بھیجیں - صحیح کیمرہ کے ساتھ
                        # ========================================
                        if weapons and self.can_send_alert(camera_id):
                            best = max(weapons, key=lambda x: x.get('confidence', 0))
                            conf = best.get('confidence', 0)
                            
                            if conf >= 0.30:
                                print(f"\n🚨 WEAPON on {camera_info['name']}!")
                                print(f"   📍 Location: {camera_info['address']}")
                                print(f"   🗺️ Coords: {camera_info['lat']}, {camera_info['lng']}")
                                
                                # 🔴 الرٹ بھیجیں - camera_info EXPLICIT
                                if self.firebase.send_alert(best, frame, camera_info):
                                    self.update_cooldown(camera_id)
                                    self.total_alerts += 1
                        
                        # ڈسپلے فریم
                        display = self.draw_boxes(frame, weapons, camera_info, fps)
                        
                        processed_frames.append({
                            'frame': display,
                            'camera_info': camera_info
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Error: {e}")
                
                # ڈسپلے
                combined = self.combine_frames(processed_frames)
                cv2.imshow("Weapon Detection | Wazirabad (Left) | Gujranwala (Right) | Press 'q'", combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.01)
            
            self.camera_handler.release_all()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\n👋 Stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"\n📊 Alerts: {self.total_alerts}")
            print(f"📊 Videos: {self.firebase.videos_uploaded}")


def main():
    print("\n" + "🎯" * 30)
    print("🎯 WEAPON DETECTION v9.0 - LOCATION FIXED")
    print("🎯" * 30)
    print("\n📍 Wazirabad (Laptop): 32.440418, 74.120255")
    print("📍 Gujranwala (Mobile): 32.187691, 74.194450\n")
    
    app = WeaponDetectionApp()
    app.run()


if __name__ == "__main__":
    main()