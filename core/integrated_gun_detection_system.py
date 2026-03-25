"""
Complete Integrated Gun Detection System with Enhanced UI

Combines YOLO gun detection with agent-based decision making
for real-time threat assessment and response.
Features a professional 4-section display with live feed, bird's eye view,
heatmap, and analytics panel.
"""

import cv2
import numpy as np
import time
import os
import platform
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from ultralytics import YOLO
from detection.human_tracker import HumanTracker
from pose_detection import PoseDetector
from fight_detection.fight_detector import ViolenceDetector
from agents.agent_based_decision_engine import AgentBasedDecisionEngine, AgentState
from explosion.fire_smoke_detection import FireSmokeDetector
from utils.alert_system import AlertSystem
from utils.firebase_alert_storage import FirebaseAlertStorage


class IntegratedGunDetectionSystem:
    """Complete integrated system with YOLO detection and agent-based decision making"""

    def __init__(self, model_path: str = "best.pt", camera_index: int = 0):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")

        # Initialize agent-based decision engine
        self.decision_engine = AgentBasedDecisionEngine()
        print("✓ Agent-based decision engine initialized")

        # Initialize human tracker with advanced activity classification
        self.human_tracker = HumanTracker()
        print("✓ Human tracker with activity classification initialized")

        # Initialize pose detector
        self.pose_detector = PoseDetector()
        print("✓ Pose detector initialized")

        # Initialize violence detector
        self.violence_detector = ViolenceDetector()
        print("✓ Violence detector initialized")

        # Initialize fire-smoke detector
        self.fire_smoke_detector = FireSmokeDetector()
        print("✓ Fire-Smoke detector initialized")

        # Initialize alert system
        self.alert_system = AlertSystem(
            camera_id="CAM_001", camera_location="Main Security Camera"
        )
        print("✓ Alert system initialized")

        # Initialize Firebase alert storage
        self.firebase_storage = FirebaseAlertStorage()
        print("✓ Firebase alert storage initialized")

        # Get reference to evidence agent for direct frame buffering
        self.evidence_agent = self.decision_engine.evidence_agent
        print("✓ Evidence agent connected for frame buffering")

        # Camera setup
        self.camera_index = camera_index
        self.cap = None

        # Tracking system
        self.person_id_counter = 0
        self.active_tracks = {}
        self.frame_count = 0

        # Evidence storage
        self.evidence_folder = "evidence"
        self.init_evidence_storage()

        # Alert system
        self.alert_active = False
        self.last_alert_time = 0

        # ═══════════════════════════════════════════════════════
        # ✅ CALLBACK SYSTEM FOR EXTERNAL HANDLERS (Firebase etc)
        # ═══════════════════════════════════════════════════════
        self._detection_callbacks = []
        self.recent_detections = []
        self.last_detections = []  # Current frame detections
        self.start_time = time.time()
        self.fps = 0.0
        self._frame_times = []

        # Statistics
        self.stats = {
            "total_detections": 0,
            "threat_detections": 0,
            "alerts_triggered": 0,
            "evidence_saved": 0,
            "hands_up_detections": 0,
            "violence_detections": 0,
            "activity_detections": {},
        }

    # ═══════════════════════════════════════════════════════════
    # ✅ NEW: CALLBACK REGISTRATION AND NOTIFICATION METHODS
    # ═══════════════════════════════════════════════════════════

    def add_detection_callback(self, callback):
        """
        Register external callback for weapon/threat detections.
        
        Callback receives dict:
        {
            'class': 'GUN',
            'confidence': 0.94,
            'bbox': [x, y, w, h],
            'type': 'WEAPON',
            'timestamp': 1772881075.5,
            'camera_id': 'CAM_001',
            'threat_level': 'HIGH',
            'detection_id': 1
        }
        """
        self._detection_callbacks.append(callback)
        print(f"✅ Detection callback registered (total: {len(self._detection_callbacks)})")

    def remove_detection_callback(self, callback):
        """Remove a registered callback"""
        if callback in self._detection_callbacks:
            self._detection_callbacks.remove(callback)
            print(f"✅ Detection callback removed")

    def _notify_callbacks(self, detection_data: Dict[str, Any]):
        """
        Notify all registered callbacks about a detection.
        Called internally whenever a weapon/threat is detected.
        """
        # Store in recent detections list
        self.recent_detections.append(detection_data)
        if len(self.recent_detections) > 100:
            self.recent_detections = self.recent_detections[-100:]

        # Update last_detections for external polling
        self.last_detections.append(detection_data)

        # Call each registered callback
        for callback in self._detection_callbacks:
            try:
                callback(detection_data)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")

    def _notify_weapon_detected(self, class_name: str, confidence: float,
                                 bbox: list = None, detection_id: int = 0,
                                 weapon_type: str = "Unknown"):
        """Notify callbacks about weapon detection"""
        detection = {
            'class': class_name.upper(),
            'confidence': float(confidence),
            'bbox': list(bbox) if bbox else None,
            'type': 'WEAPON',
            'weapon_type': weapon_type,
            'timestamp': time.time(),
            'camera_id': 'CAM_001',
            'location': 'Main Security Camera',
            'threat_level': 'HIGH' if confidence > 0.7 else 'MEDIUM',
            'detection_id': detection_id,
            'frame_count': self.frame_count
        }
        self._notify_callbacks(detection)

    def _notify_violence_detected(self, person_id: int, confidence: float,
                                   bbox: list = None):
        """Notify callbacks about violence detection"""
        detection = {
            'class': 'VIOLENCE',
            'confidence': float(confidence),
            'bbox': list(bbox) if bbox else None,
            'type': 'VIOLENCE',
            'timestamp': time.time(),
            'camera_id': 'CAM_001',
            'location': 'Main Security Camera',
            'threat_level': 'HIGH' if confidence > 0.7 else 'MEDIUM',
            'person_id': person_id,
            'frame_count': self.frame_count
        }
        self._notify_callbacks(detection)

    def _notify_fire_smoke_detected(self, detection_type: str,
                                     confidence: float, count: int = 1,
                                     bbox: list = None):
        """Notify callbacks about fire/smoke detection"""
        detection = {
            'class': detection_type.upper(),
            'confidence': float(confidence),
            'bbox': list(bbox) if bbox else None,
            'type': detection_type.upper(),
            'timestamp': time.time(),
            'camera_id': 'CAM_001',
            'location': 'Main Security Camera',
            'threat_level': 'CRITICAL',
            'count': count,
            'frame_count': self.frame_count
        }
        self._notify_callbacks(detection)

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self._frame_times.append(current_time)
        # Keep last 30 frame times
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.fps = (len(self._frame_times) - 1) / elapsed

    # ═══════════════════════════════════════════════════════════
    # EXISTING METHODS (UPDATED)
    # ═══════════════════════════════════════════════════════════

    def init_evidence_storage(self):
        """Initialize evidence storage folder and database"""
        os.makedirs(self.evidence_folder, exist_ok=True)

        self.db_path = os.path.join(self.evidence_folder, "detections.db")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                detection_id TEXT,
                bbox TEXT,
                confidence REAL,
                threat_level TEXT,
                threat_score REAL,
                actions TEXT,
                evidence_path TEXT,
                frame_data BLOB
            )
        """
        )

        conn.commit()
        conn.close()
        print(f"✓ Evidence storage initialized: {self.evidence_folder}")

    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        print(f"✓ Camera {self.camera_index} started")
        return True

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], Dict]:
        """Detect objects using YOLO model"""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = self.model(frame, stream=True, conf=0.3)
        weapon_detections = []

        results_list = list(results)
        print(f"🔍 YOLO Results: {len(results_list)} result(s)")

        for r in results_list:
            boxes = r.boxes
            print(f"📦 Boxes in result: {len(boxes)}")
            for box in boxes:
                detection = self.convert_yolo_to_detection(box, frame)
                if detection:
                    weapon_detections.append(detection)

        # Detect humans
        human_detections = self.human_tracker.detect_humans(frame)

        # Detect fire and smoke
        fire_smoke_result = self.fire_smoke_detector.detect_fire_smoke_in_frame(frame)

        # Combine detections
        all_detections = weapon_detections + human_detections

        # ✅ NOTIFY CALLBACKS: Fire/Smoke detected
        if fire_smoke_result["fire_detected"]:
            print(
                f"🔥 EMERGENCY: Fire ({fire_smoke_result['fire_count']}) detected!"
            )
            self._notify_fire_smoke_detected(
                "FIRE",
                fire_smoke_result.get("fire_confidence", 0.9),
                fire_smoke_result["fire_count"],
            )

        if fire_smoke_result["smoke_detected"]:
            print(
                f"💨 EMERGENCY: Smoke ({fire_smoke_result['smoke_count']}) detected!"
            )
            self._notify_fire_smoke_detected(
                "SMOKE",
                fire_smoke_result.get("smoke_confidence", 0.8),
                fire_smoke_result["smoke_count"],
            )

        return all_detections, fire_smoke_result
    
    def process_frame(self, frame):
        """
        Process a single frame - wrapper for detection
        This method is called by external apps to get frame detections
        """
        if frame is None:
            return None
        
        try:
            # Detect objects using existing method
            detections, fire_smoke_result = self.detect_objects(frame)
            
            # Process detections through agent-based decision engine
            results = self.process_detections(detections, frame)
            
            # Generate alerts for Firebase/Cloudinary
            self.generate_detection_alerts(detections, fire_smoke_result, results)
            
            # Update frame count
            self.frame_count += 1
            
            # Optional: Update FPS
            self._update_fps()
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_yolo_to_detection(
        self, box, frame: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Convert YOLO detection to agent format"""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            original_class = self.model.names[cls].upper()

            # Enhanced weapon classification
            if original_class == "GUN":
                class_name = "GUN"
                weapon_type = "Firearm"
            elif original_class == "KNIFE":
                class_name = "KNIFE"
                weapon_type = "Blade Weapon"
            elif original_class == "EXPLOSION":
                class_name = "EXPLOSION"
                weapon_type = "Explosive Threat"
            elif original_class == "GRENADE":
                class_name = "GRENADE"
                weapon_type = "Explosive Device"
            else:
                class_name = original_class
                weapon_type = "Unknown Object"

            print(f"🔍 DETECTION: {original_class} with confidence {conf:.2f}")

            # ═══════════════════════════════════════════════
            # ✅ WEAPON DETECTED → NOTIFY ALL CALLBACKS
            # ═══════════════════════════════════════════════
            if class_name in ["GUN", "KNIFE", "EXPLOSION", "GRENADE"]:
                print(
                    f"🎯 WEAPON DETECTED: {class_name} with confidence {conf:.2f}"
                )
                # Create or update track first to get person_id
                person_id = self.update_track(x1, y1, w, h, class_name, conf)

                # ✅ NOTIFY EXTERNAL CALLBACKS (Firebase Realtime DB etc)
                self._notify_weapon_detected(
                    class_name=class_name,
                    confidence=conf,
                    bbox=[x1, y1, w, h],
                    detection_id=person_id,
                    weapon_type=weapon_type,
                )
            else:
                person_id = self.update_track(x1, y1, w, h, class_name, conf)

            # Build detection dict
            detection = {
                "id": person_id,
                "bbox": [x1, y1, w, h],
                "person_conf": conf if class_name == "GUN" else 0.0,
                "gun_conf": conf if class_name == "GUN" else 0.0,
                "knife_conf": conf if class_name == "KNIFE" else 0.0,
                "explosion_conf": conf if class_name == "EXPLOSION" else 0.0,
                "grenade_conf": conf if class_name == "GRENADE" else 0.0,
                "violence_detected": False,
                "violence_confidence": 0.0,
                "meta": {
                    "class_name": class_name,
                    "weapon_type": weapon_type,
                    "raw_confidence": conf,
                    "frame": self.frame_count,
                    "camera": self.camera_index,
                },
                "timestamp": time.time(),
                "frame": frame.copy(),
            }

            return detection

        except Exception as e:
            print(f"Error converting detection: {e}")
            return None

    def update_track(
        self, x: int, y: int, w: int, h: int, class_name: str, conf: float
    ) -> int:
        """Update or create person tracks"""
        center_x, center_y = x + w // 2, y + h // 2

        best_match_id = None
        best_distance = float("inf")

        for track_id, track_info in self.active_tracks.items():
            track_center = track_info["center"]
            distance = np.sqrt(
                (center_x - track_center[0]) ** 2
                + (center_y - track_center[1]) ** 2
            )

            if distance < 100 and distance < best_distance:
                best_distance = distance
                best_match_id = track_id

        if best_match_id is None:
            self.person_id_counter += 1
            best_match_id = self.person_id_counter

        self.active_tracks[best_match_id] = {
            "center": (center_x, center_y),
            "bbox": [x, y, w, h],
            "last_seen": self.frame_count,
            "class_name": class_name,
            "confidence": conf,
        }

        self.clean_old_tracks()
        return best_match_id

    def clean_old_tracks(self):
        """Remove tracks not seen recently with occlusion handling"""
        current_frame = self.frame_count
        stale_ids = []

        for track_id, track_info in self.active_tracks.items():
            frames_since_last_seen = current_frame - track_info["last_seen"]

            if 3 <= frames_since_last_seen <= 6:
                track_info["occluded"] = True
                track_info["occlusion_count"] = (
                    track_info.get("occlusion_count", 0) + 1
                )
                print(
                    f"👁️ Person {track_id} potentially occluded for "
                    f"{frames_since_last_seen} frames"
                )
            elif frames_since_last_seen > 6:
                stale_ids.append(track_id)
                print(
                    f"🗑️ Removing stale track {track_id} "
                    f"(not seen for {frames_since_last_seen} frames)"
                )
            else:
                track_info["occluded"] = False
                track_info["occlusion_count"] = 0

        for stale_id in stale_ids:
            del self.active_tracks[stale_id]

    def process_detections(
        self, detections: List[Dict[str, Any]], frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Process detections through agent-based decision engine"""
        results = []

        # Clear last_detections for this frame
        self.last_detections = []

        if not hasattr(self, "detection_history"):
            self.detection_history = []
        self.detection_history.append(detections)
        self.detection_history = self.detection_history[-100:]

        # Add frame to evidence buffer
        self.evidence_agent.add_frame_to_buffer(frame, time.time())

        # Detect poses for all persons in frame
        person_detections = [
            d for d in detections if d.get("meta", {}).get("class_name") == "PERSON"
        ]
        if person_detections:
            pose_results = self.pose_detector.detect_poses_in_frame(
                frame, person_detections
            )

            hands_up_count = self.pose_detector.get_hands_up_count()
            if hands_up_count > self.stats.get("hands_up_detections", 0):
                self.stats["hands_up_detections"] = hands_up_count
                hands_up_ids = self.pose_detector.get_hands_up_person_ids()
                print(
                    f"🙋 HANDS-UP DETECTED: {hands_up_count} persons "
                    f"(IDs: {hands_up_ids})"
                )

            violence_results = self.violence_detector.detect_violence_in_frame(
                frame, person_detections
            )

            violence_count = self.violence_detector.get_violence_count()
            if violence_count > self.stats.get("violence_detections", 0):
                self.stats["violence_detections"] = violence_count
                violent_ids = self.violence_detector.get_violent_person_ids()
                print(
                    f"🥊 VIOLENCE DETECTED: {violence_count} persons "
                    f"(IDs: {violent_ids})"
                )

                for person_id in violent_ids:
                    violence_info = self.violence_detector.get_violence_info(
                        person_id
                    )
                    if violence_info and violence_info.get(
                        "violence_detected", False
                    ):
                        # ✅ NOTIFY CALLBACKS: Violence detected
                        self._notify_violence_detected(
                            person_id=person_id,
                            confidence=violence_info.get("confidence", 0.0),
                            bbox=violence_info.get("bbox"),
                        )

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = (
                            f"violence_evidence_{person_id}_{timestamp}.jpg"
                        )
                        filepath = f"{self.evidence_folder}/{filename}"

                        annotated_frame = self.draw_violence_evidence(
                            frame, violence_info
                        )
                        cv2.imwrite(filepath, annotated_frame)

                        self.stats["evidence_saved"] += 1
                        print(f"✓ Violence evidence saved: {filename}")

        for detection in detections:
            person_id = detection.get("id")
            if person_id and person_id in self.pose_detector.detected_poses:
                pose_info = self.pose_detector.detected_poses[person_id]
                detection["pose_type"] = pose_info.get("pose_type", "NORMAL")
                detection["pose_confidence"] = pose_info.get(
                    "confidence", 0.0
                )
                detection["pose_keypoints"] = pose_info.get("keypoints", [])

            if (
                person_id
                and person_id in self.violence_detector.detected_violence
            ):
                violence_info = self.violence_detector.detected_violence[
                    person_id
                ]
                detection["violence_detected"] = violence_info.get(
                    "violence_detected", False
                )
                detection["violence_confidence"] = violence_info.get(
                    "confidence", 0.0
                )

            result = self.decision_engine.process(detection)
            results.append(result)

            self.stats["total_detections"] += 1
            if result["threat_score"] > 1.0:
                self.stats["threat_detections"] += 1

            if "SAVE_EVIDENCE" in result["action"]:
                self.save_evidence(detection, result, frame)

            if "LOCAL_ALARM" in result["action"]:
                self.trigger_alert(result)

            self.save_to_database(detection, result, frame)

        return results

    def save_evidence(
        self,
        detection: Dict[str, Any],
        result: Dict[str, Any],
        frame: np.ndarray,
    ):
        """Save evidence frame"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_{detection['id']}_{timestamp}.jpg"
            filepath = f"{self.evidence_folder}/{filename}"

            annotated_frame = self.draw_annotations(frame, detection, result)
            cv2.imwrite(filepath, annotated_frame)

            self.stats["evidence_saved"] += 1
            print(f"✓ Evidence saved: {filename}")

        except Exception as e:
            print(f"Error saving evidence: {e}")

    def trigger_alert(self, result: Dict[str, Any]):
        """Trigger alert system"""
        current_time = time.time()

        if current_time - self.last_alert_time < 2:
            return

        self.last_alert_time = current_time
        self.stats["alerts_triggered"] += 1

        self.play_alert_sound(result["state"])
        print(
            f"🚨 ALERT TRIGGERED: {result['state']} "
            f"(Score: {result['threat_score']:.2f})"
        )

    def play_alert_sound(self, threat_level: str):
        """Play alert sound based on threat level"""
        try:
            if platform.system() == "Windows":
                import winsound

                if threat_level in ["CRITICAL", "VIOLENT"]:
                    winsound.Beep(1500, 500)
                elif threat_level in ["HIGH", "ARMED"]:
                    winsound.Beep(1000, 300)
                else:
                    winsound.Beep(800, 200)
            else:
                print("\a")
        except Exception as e:
            print(f"Alert sound failed: {e}")

    def save_to_database(
        self,
        detection: Dict[str, Any],
        result: Dict[str, Any],
        frame: np.ndarray,
    ):
        """Save detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            _, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            frame_data = buffer.tobytes()

            cursor.execute(
                """
                INSERT INTO detections 
                (timestamp, detection_id, bbox, confidence, threat_level,
                 threat_score, actions, evidence_path, frame_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    detection["timestamp"],
                    detection["id"],
                    json.dumps(detection["bbox"]),
                    detection.get("person_conf", 0),
                    result["state"],
                    result["threat_score"],
                    result["action"],
                    f"evidence_{detection['id']}.jpg",
                    frame_data,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Database error: {e}")

    def draw_violence_evidence(
        self, frame: np.ndarray, violence_info: Dict[str, Any]
    ) -> np.ndarray:
        """Draw violence evidence on frame with red bounding box"""
        try:
            annotated = frame.copy()
            person_id = violence_info.get("person_id")
            violence_detected = violence_info.get("violence_detected", False)
            confidence = violence_info.get("confidence", 0.0)
            bbox = violence_info.get("bbox", [])

            if not bbox or len(bbox) < 4:
                return annotated

            x, y, w, h = bbox[:4]

            if violence_detected:
                cv2.rectangle(
                    annotated, (x, y), (x + w, y + h), (0, 0, 255), 4
                )

                evidence_label = f"VIOLENCE DETECTED - ID:{person_id}"
                conf_text = f"Confidence: {confidence:.2f}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                label_size = cv2.getTextSize(
                    evidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )[0]
                cv2.rectangle(
                    annotated,
                    (x, y - 80),
                    (x + max(label_size[0] + 20, 400), y - 20),
                    (0, 0, 255),
                    -1,
                )
                cv2.rectangle(
                    annotated,
                    (x, y - 80),
                    (x + max(label_size[0] + 20, 400), y - 20),
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated,
                    evidence_label,
                    (x + 10, y - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    conf_text,
                    (x + 10, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated,
                    timestamp,
                    (x + 10, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    annotated,
                    "EVIDENCE RECORDING",
                    (annotated.shape[1] // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

            return annotated

        except Exception as e:
            print(f"Error drawing violence evidence: {e}")
            return frame

    def draw_dashed_rectangle(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ):
        """Draw a dashed rectangle for occluded persons"""
        x1, y1 = pt1
        x2, y2 = pt2

        for x in range(x1, x2, dash_length * 2):
            cv2.line(
                frame,
                (x, y1),
                (min(x + dash_length, x2), y1),
                color,
                thickness,
            )
        for x in range(x1, x2, dash_length * 2):
            cv2.line(
                frame,
                (x, y2),
                (min(x + dash_length, x2), y2),
                color,
                thickness,
            )
        for y in range(y1, y2, dash_length * 2):
            cv2.line(
                frame,
                (x1, y),
                (x1, min(y + dash_length, y2)),
                color,
                thickness,
            )
        for y in range(y1, y2, dash_length * 2):
            cv2.line(
                frame,
                (x2, y),
                (x2, min(y + dash_length, y2)),
                color,
                thickness,
            )

    def draw_annotations(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        result: Dict[str, Any],
    ) -> np.ndarray:
        """Draw professional annotations on frame with occlusion handling"""
        annotated = frame.copy()

        bbox = detection["bbox"]
        x, y, w, h = bbox
        person_id = detection["id"]

        is_occluded = False
        occlusion_count = 0
        if person_id in self.active_tracks:
            track_info = self.active_tracks[person_id]
            is_occluded = track_info.get("occluded", False)
            occlusion_count = track_info.get("occlusion_count", 0)

        system_state = result.get("system_state", "normal").upper()
        color = self.get_system_state_color(system_state)

        if is_occluded:
            self.draw_dashed_rectangle(
                annotated, (x, y), (x + w, y + h), color, 2
            )
            occlusion_label = (
                f"ID:{person_id} OCCLUDED ({occlusion_count}x)"
            )
            label_size = cv2.getTextSize(
                occlusion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.rectangle(
                annotated,
                (x, y - 30),
                (x + label_size[0] + 10, y - 10),
                (255, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                occlusion_label,
                (x + 5, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        else:
            for i in range(3):
                cv2.rectangle(
                    annotated,
                    (x - i, y - i),
                    (x + w + i, y + h + i),
                    (color[0], color[1], color[2]),
                    1,
                )
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        label = f"ID:{detection['id']} {system_state}"
        score_text = f"Score:{result['threat_score']:.1f}"

        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0]
        bg_height = 60
        cv2.rectangle(
            annotated,
            (x, y - bg_height),
            (x + label_size[0] + 20, y),
            color,
            -1,
        )
        cv2.rectangle(
            annotated,
            (x, y - bg_height),
            (x + label_size[0] + 20, y),
            (255, 255, 255),
            1,
        )

        cv2.putText(
            annotated,
            label,
            (x + 10, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            score_text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

        violence_detected = detection.get("violence_detected", False)
        violence_confidence = detection.get("violence_confidence", 0.0)

        if violence_detected:
            violence_color = (0, 0, 255)
            violence_label = f"VIOLENCE ID:{detection['id']}"
            violence_conf_text = f"Violence:{violence_confidence:.2f}"

            violence_label_size = cv2.getTextSize(
                violence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )[0]
            cv2.rectangle(
                annotated,
                (x, y - 105),
                (x + violence_label_size[0] + 10, y - 80),
                violence_color,
                -1,
            )
            cv2.putText(
                annotated,
                violence_label,
                (x + 5, y - 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                violence_conf_text,
                (x + 5, y - 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                violence_color,
                1,
            )

            cv2.putText(
                annotated,
                "VIOLENCE DETECTED",
                (x, y - 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                violence_color,
                2,
            )

            violence_info = {
                "person_id": detection["id"],
                "violence_detected": violence_detected,
                "confidence": violence_confidence,
                "bbox": bbox,
            }
            annotated = self.violence_detector.draw_violence_on_frame(
                annotated, violence_info
            )

        pose_type = detection.get("pose_type", "NORMAL")
        pose_confidence = detection.get("pose_confidence", 0.0)

        if pose_type == "HANDS_UP":
            pose_color = (0, 255, 255)
            pose_label = "HANDS_UP"
            pose_conf_text = f"Pose:{pose_confidence:.2f}"

            pose_label_size = cv2.getTextSize(
                pose_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.rectangle(
                annotated,
                (x, y - 85),
                (x + pose_label_size[0] + 10, y - 60),
                pose_color,
                -1,
            )
            cv2.putText(
                annotated,
                pose_label,
                (x + 5, y - 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                pose_conf_text,
                (x + 5, y - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                pose_color,
                1,
            )

            pose_keypoints = detection.get("pose_keypoints", [])
            if pose_keypoints:
                pose_info = {
                    "person_id": detection["id"],
                    "pose_type": pose_type,
                    "confidence": pose_confidence,
                    "keypoints": pose_keypoints,
                    "bbox": bbox,
                }
                annotated = self.pose_detector.draw_pose_on_frame(
                    annotated, pose_info, pose_color
                )

        confidences = []
        if detection.get("gun_conf", 0) > 0.1:
            confidences.append(f"🔫 Gun:{detection['gun_conf']:.2f}")
        if detection.get("knife_conf", 0) > 0.1:
            confidences.append(f"🔪 Knife:{detection['knife_conf']:.2f}")
        if detection.get("explosion_conf", 0) > 0.1:
            confidences.append(
                f"💥 Explosion:{detection['explosion_conf']:.2f}"
            )
        if detection.get("grenade_conf", 0) > 0.1:
            confidences.append(f"🧨 Grenade:{detection['grenade_conf']:.2f}")

        explosion_conf = detection.get("explosion_conf", 0)
        grenade_conf = detection.get("grenade_conf", 0)

        if explosion_conf > 0.3 or grenade_conf > 0.3:
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (0, 0, 255), 5
            )

            if explosion_conf > grenade_conf:
                explosive_label = "💥 EXPLOSION DETECTED"
                conf_text = f"Explosion:{explosion_conf:.2f}"
            else:
                explosive_label = "🧨 GRENADE DETECTED"
                conf_text = f"Grenade:{grenade_conf:.2f}"

            label_size = cv2.getTextSize(
                explosive_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            cv2.rectangle(
                annotated,
                (x, y - 50),
                (x + max(label_size[0] + 20, 350), y - 10),
                (0, 0, 255),
                -1,
            )
            cv2.rectangle(
                annotated,
                (x, y - 50),
                (x + max(label_size[0] + 20, 350), y - 10),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                explosive_label,
                (x + 10, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                conf_text,
                (x + 10, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                annotated,
                "CRITICAL THREAT!",
                (annotated.shape[1] // 2 - 100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        if confidences:
            conf_text = " | ".join(confidences[:3])
            cv2.putText(
                annotated,
                conf_text,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        if result.get("state_changed", False):
            cv2.putText(
                annotated,
                "STATE CHANGED!",
                (x, y + h + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        emergency_response = result.get("emergency_response")
        if emergency_response and not emergency_response.get(
            "emergency_deactivated"
        ):
            threat_type = emergency_response.get("threat_type", "UNKNOWN")
            cv2.putText(
                annotated,
                f"EMERGENCY: {threat_type}",
                (annotated.shape[1] // 2 - 150, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        return annotated

    def get_system_state_color(self, state: str) -> tuple:
        """Get color based on system state"""
        colors = {
            "NORMAL": (0, 255, 0),
            "SUSPICIOUS": (0, 255, 255),
            "THREAT_DETECTION": (0, 165, 255),
            "EMERGENCY": (0, 0, 255),
            "MINIMAL": (0, 255, 0),
            "LOW": (255, 255, 0),
            "MEDIUM": (255, 165, 0),
            "HIGH": (255, 0, 0),
            "CRITICAL": (255, 0, 0),
            "VIOLENT": (0, 0, 255),
            "ARMED": (255, 165, 0),
        }
        return colors.get(state, (255, 255, 255))

    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw professional statistics on frame"""
        stats_frame = frame.copy()
        height, width = stats_frame.shape[:2]

        cv2.rectangle(stats_frame, (0, 0), (width, 60), (20, 20, 20), -1)
        cv2.rectangle(stats_frame, (0, 0), (width, 60), (0, 255, 0), 2)

        cv2.putText(
            stats_frame,
            "INTELLIGENT WEAPON DETECTION SYSTEM",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            stats_frame,
            "AI-Powered Security Monitoring",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        panel_width = 200
        cv2.rectangle(
            stats_frame,
            (width - panel_width, 60),
            (width, height - 40),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            stats_frame,
            (width - panel_width, 60),
            (width, height - 40),
            (0, 255, 0),
            2,
        )

        current_state = getattr(
            self.decision_engine.state_agent, "state_transition", None
        )
        system_state = (
            current_state.current_state.value
            if current_state
            else "UNKNOWN"
        )
        state_color = self.get_system_state_color(system_state.upper())

        cv2.putText(
            stats_frame,
            "STATUS",
            (width - panel_width + 10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            stats_frame,
            (width - panel_width + 10, 90),
            (width - panel_width + 25, 105),
            state_color,
            -1,
        )
        cv2.putText(
            stats_frame,
            f"{system_state[:4].upper()}",
            (width - panel_width + 30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            state_color,
            2,
        )

        stats_text = [
            f"Total: {self.stats['total_detections']}",
            f"Threat: {self.stats['threat_detections']}",
            f"Alerts: {self.stats['alerts_triggered']}",
            f"Evidence: {self.stats['evidence_saved']}",
            "━━━━━━━━━━━━",
            f"FPS: {self.fps:.1f}",
        ]

        y_offset = 125
        for text in stats_text:
            color = (
                (255, 255, 255)
                if text.startswith(("Total", "Threat", "Alerts", "Evidence"))
                else (100, 100, 100)
            )
            if text.startswith("FPS:"):
                color = (0, 255, 0)
            cv2.putText(
                stats_frame,
                text,
                (width - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
            y_offset += 18

        cv2.rectangle(
            stats_frame, (0, height - 40), (width, height), (20, 20, 20), -1
        )
        cv2.rectangle(
            stats_frame, (0, height - 40), (width, height), (0, 255, 0), 1
        )

        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence [W] Reset Recording"
        cv2.putText(
            stats_frame,
            controls,
            (20, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            stats_frame,
            timestamp,
            (width - 100, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        return stats_frame

    def create_birds_eye_view(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create bird's eye view of detection area"""
        birds_eye = np.zeros((240, 320, 3), dtype=np.uint8)
        birds_eye.fill(20)

        for i in range(0, 320, 40):
            cv2.line(birds_eye, (i, 0), (i, 240), (40, 40, 40), 1)
        for i in range(0, 240, 40):
            cv2.line(birds_eye, (0, i), (320, i), (40, 40, 40), 1)

        cv2.rectangle(birds_eye, (10, 10), (310, 230), (0, 255, 0), 2)

        cv2.putText(
            birds_eye,
            "BIRD'S EYE VIEW",
            (80, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                bird_x = int((x + w / 2) * 320 / frame.shape[1])
                bird_y = int((y + h / 2) * 240 / frame.shape[0])

                threat_score = detection.get("threat_score", 0)
                if threat_score > 2.0:
                    color = (0, 0, 255)
                elif threat_score > 1.0:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                cv2.circle(birds_eye, (bird_x, bird_y), 8, color, -1)
                cv2.circle(birds_eye, (bird_x, bird_y), 10, color, 2)

                cv2.putText(
                    birds_eye,
                    str(detection.get("id", "?")),
                    (bird_x - 5, bird_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

        return birds_eye

    def create_enhanced_heatmap(
        self,
        frame: np.ndarray,
        detections_history: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Create enhanced activity heatmap"""
        heatmap = np.zeros((180, 240, 3), dtype=np.uint8)
        heatmap.fill(20)

        cv2.putText(
            heatmap,
            "ACTIVITY HEATMAP",
            (50, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        if detections_history:
            grid_size = 20
            for i in range(0, 240, grid_size):
                cv2.line(heatmap, (i, 30), (i, 180), (30, 30, 30), 1)
            for i in range(30, 180, grid_size):
                cv2.line(heatmap, (0, i), (240, i), (30, 30, 30), 1)

            activity_grid = np.zeros((8, 12), dtype=float)

            for detection_batch in detections_history[-100:]:
                if isinstance(detection_batch, list):
                    for detection in detection_batch:
                        if isinstance(detection, dict):
                            bbox = detection.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                x, y, w, h = bbox[:4]
                                center_x = int(
                                    (x + w / 2) / frame.shape[1] * 12
                                )
                                center_y = int(
                                    (y + h / 2) / frame.shape[0] * 8
                                )

                                if 0 <= center_x < 12 and 0 <= center_y < 8:
                                    threat_score = detection.get(
                                        "threat_score", 0.5
                                    )
                                    activity_grid[center_y, center_x] += (
                                        threat_score
                                    )

            try:
                from scipy.ndimage import gaussian_filter

                activity_grid = gaussian_filter(activity_grid, sigma=1.0)
            except ImportError:
                pass

            max_activity = (
                np.max(activity_grid)
                if np.max(activity_grid) > 0
                else 1
            )

            for y in range(8):
                for x in range(12):
                    intensity = activity_grid[y, x] / max_activity

                    if intensity < 0.2:
                        color = (0, 0, int(intensity * 5 * 255))
                    elif intensity < 0.4:
                        ratio = (intensity - 0.2) / 0.2
                        color = (0, int(ratio * 255), 255)
                    elif intensity < 0.6:
                        ratio = (intensity - 0.4) / 0.2
                        color = (0, 255, int((1 - ratio) * 255))
                    elif intensity < 0.8:
                        ratio = (intensity - 0.6) / 0.2
                        color = (int(ratio * 255), 255, 0)
                    else:
                        ratio = (intensity - 0.8) / 0.2
                        color = (255, int((1 - ratio) * 255), 0)

                    cell_x = x * grid_size
                    cell_y = 30 + y * grid_size

                    cv2.rectangle(
                        heatmap,
                        (cell_x, cell_y),
                        (cell_x + grid_size, cell_y + grid_size),
                        color,
                        -1,
                    )
                    cv2.rectangle(
                        heatmap,
                        (cell_x, cell_y),
                        (cell_x + grid_size, cell_y + grid_size),
                        (color[0] // 2, color[1] // 2, color[2] // 2),
                        1,
                    )

                    if activity_grid[y, x] > 0.5:
                        count = int(activity_grid[y, x])
                        cv2.putText(
                            heatmap,
                            str(count),
                            (cell_x + 5, cell_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1,
                        )

        return heatmap

    def get_activity_statistics(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get activity statistics from detections"""
        activity_stats = {
            "Sitting": 0,
            "Standing": 0,
            "Walking": 0,
            "Running": 0,
            "Lying": 0,
            "HandsUp": 0,
            "Aiming": 0,
            "Unknown": 0,
        }

        for detection in detections:
            if (
                detection.get("meta", {}).get("class_name") == "PERSON"
            ):
                activity = detection.get("meta", {}).get(
                    "activity", "Unknown"
                )
                if activity in activity_stats:
                    activity_stats[activity] += 1
                else:
                    activity_stats["Unknown"] += 1

        return activity_stats

    def get_enhanced_system_state(
        self, detections: List[Dict[str, Any]]
    ) -> str:
        """Enhanced state management based on weapon and pose combinations"""
        has_weapon = False
        has_aiming = False
        has_hands_up = False
        weapon_count = 0
        aiming_count = 0
        hands_up_count = 0

        for detection in detections:
            class_name = (
                detection.get("meta", {}).get("class_name", "").upper()
            )
            activity = detection.get("meta", {}).get("activity", "Unknown")

            if class_name in ["GUN", "KNIFE", "WEAPON"]:
                has_weapon = True
                weapon_count += 1

            if activity.upper() in ["AIMING", "HANDSUP"]:
                if activity.upper() == "AIMING":
                    has_aiming = True
                    aiming_count += 1
                elif activity.upper() == "HANDSUP":
                    has_hands_up = True
                    hands_up_count += 1

        if has_weapon and (has_aiming or has_hands_up):
            print(
                f"🚨 EMERGENCY STATE: Weapon ({weapon_count}) + "
                f"Pose (Aiming: {aiming_count}, HandsUp: {hands_up_count})"
            )
            self.trigger_emergency_alert()
            return "EMERGENCY"
        elif has_weapon:
            print(f"⚠️ WEAPON DETECTED: {weapon_count} weapons found")
            return "WEAPON_DETECTED"
        elif has_aiming or has_hands_up:
            print(
                f"🔍 SUSPICIOUS: Aiming ({aiming_count}) or "
                f"HandsUp ({hands_up_count})"
            )
            return "SUSPICIOUS"
        else:
            return "NORMAL"

    def trigger_emergency_alert(self):
        """Trigger emergency alert with beep"""
        try:
            import winsound

            winsound.Beep(1000, 500)
            print("🚨 EMERGENCY ALERT BEEP ACTIVATED!")
        except ImportError:
            print("🚨 EMERGENCY ALERT (Visual Only)")
        except Exception as e:
            print(f"🚨 EMERGENCY ALERT (Sound Error: {e})")

    def get_person_state_info(
        self, detection: Dict[str, Any]
    ) -> tuple:
        """Get individual person's state information"""
        person_id = detection.get("id", 0)
        activity = detection.get("meta", {}).get("activity", "Unknown")
        class_name = (
            detection.get("meta", {}).get("class_name", "").upper()
        )

        person_state = "NORMAL"
        if class_name in ["GUN", "KNIFE", "WEAPON"]:
            person_state = "WEAPON_DETECTED"
        elif activity.upper() in ["AIMING", "HANDSUP"]:
            person_state = "SUSPICIOUS"

        return person_id, activity, person_state

    def create_vertical_analytics_panel(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Create professional analytics panel with agent insights"""
        analytics = np.zeros((540, 240, 3), dtype=np.uint8)
        analytics.fill(20)

        cv2.putText(
            analytics,
            "THREAT ANALYTICS",
            (50, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        system_state = (
            self.get_enhanced_system_state(detections)
            if detections
            else "NORMAL"
        )

        self.current_detections = detections
        current_people = len(
            [
                d
                for d in (detections or [])
                if d.get("meta", {}).get("class_name") == "PERSON"
            ]
        )
        all_person_ids = set()
        if hasattr(self, "detection_history"):
            for detection_batch in self.detection_history[-100:]:
                if isinstance(detection_batch, list):
                    for detection in detection_batch:
                        if (
                            isinstance(detection, dict)
                            and detection.get("meta", {}).get("class_name")
                            == "PERSON"
                        ):
                            all_person_ids.add(detection.get("id", 0))
        total_unique_people = len(all_person_ids)

        agent_insights = self._get_agent_insights(detections)

        analytics_data = [
            (
                "System State:",
                system_state.upper(),
                self.get_system_state_color(system_state.upper()),
            ),
            ("", "", (255, 255, 255)),
            ("👥 People Tracking", "", (0, 255, 0)),
            ("Current:", str(current_people), (0, 255, 255)),
            ("Total Unique:", str(total_unique_people), (0, 255, 255)),
            ("", "", (255, 255, 255)),
            ("🤖 Agent Insights", "", (255, 165, 0)),
            (
                "Threat Score:",
                f"{agent_insights.get('max_threat_score', 0):.1f}",
                (255, 165, 0),
            ),
            (
                "Dominant Pose:",
                agent_insights.get("dominant_pose", "Unknown"),
                (0, 255, 255),
            ),
            (
                "Patterns:",
                str(agent_insights.get("patterns_count", 0)),
                (255, 255, 0),
            ),
            (
                "Memory Contexts:",
                str(agent_insights.get("memory_contexts", 0)),
                (0, 255, 0),
            ),
            ("", "", (255, 255, 255)),
            ("🔍 Threat Status", "", (255, 165, 0)),
            (
                "Total Detections:",
                str(self.stats["total_detections"]),
                (255, 255, 255),
            ),
            (
                "Threat Detections:",
                str(self.stats["threat_detections"]),
                (255, 165, 0),
            ),
            (
                "Alerts Triggered:",
                str(self.stats["alerts_triggered"]),
                (255, 0, 0),
            ),
            ("", "", (255, 255, 255)),
            ("⚡ System Status", "", (0, 255, 0)),
            (
                "FPS:",
                f"{self.fps:.1f}",
                (0, 255, 0),
            ),
            (
                "CPU:",
                f"{agent_insights.get('cpu', 45)}%",
                (255, 255, 0),
            ),
            (
                "Memory:",
                f"{agent_insights.get('memory', 512)}MB",
                (255, 165, 0),
            ),
            ("", "", (255, 255, 255)),
            ("📊 Session Info", "", (0, 255, 255)),
            (
                "Uptime:",
                agent_insights.get("uptime", "00:00:00"),
                (0, 255, 255),
            ),
            (
                "Evidence Files:",
                str(len([f for f in os.listdir("evidence/videos") if f.endswith(".mp4")])),
                (0, 255, 255),
            ),
            ("", "", (255, 255, 255)),
            ("📡 Callback Status", "", (255, 165, 0)),
            (
                "Callbacks:",
                str(len(self._detection_callbacks)),
                (0, 255, 255),
            ),
            (
                "RT Alerts:",
                str(len(self.recent_detections)),
                (255, 165, 0),
            ),
        ]

        y_offset = 50
        for label, value, color in analytics_data:
            if label:
                if label.startswith(("👥", "🤖", "🔍", "⚡", "📊", "📡")):
                    cv2.putText(
                        analytics,
                        label,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
                else:
                    cv2.putText(
                        analytics,
                        label,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (200, 200, 200),
                        1,
                    )
                    cv2.putText(
                        analytics,
                        value,
                        (120, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                    )
            y_offset += 18

        cv2.putText(
            analytics,
            "AI-POWERED SECURITY",
            (50, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (100, 100, 100),
            1,
        )

        return analytics

    def _get_agent_insights(
        self, detections: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get insights from enhanced agents"""
        insights = {
            "max_threat_score": 0.0,
            "dominant_pose": "Unknown",
            "patterns_count": 0,
            "memory_contexts": 0,
            "fps": self.fps,
            "cpu": 45,
            "memory": 512,
            "uptime": "00:00:00",
        }

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        insights["uptime"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        insights["fps"] = self.fps

        if not detections:
            return insights

        try:
            if hasattr(self.decision_engine, "threat_agent"):
                for detection in detections:
                    try:
                        threat_assessment = (
                            self.decision_engine.threat_agent._assess_immediate_threat(
                                self.decision_engine.threat_agent._create_threat_context(
                                    detection, {}
                                )
                            )
                        )
                        threat_score = threat_assessment.get("level", 0)
                        insights["max_threat_score"] = max(
                            insights["max_threat_score"], threat_score
                        )
                    except Exception:
                        pass

            pose_counts = {}
            for detection in detections:
                activity = detection.get("meta", {}).get(
                    "activity", "Unknown"
                )
                pose_counts[activity] = pose_counts.get(activity, 0) + 1

            if pose_counts:
                insights["dominant_pose"] = max(
                    pose_counts.items(), key=lambda x: x[1]
                )[0]

            if hasattr(self.decision_engine, "memory_agent"):
                insights["memory_contexts"] = len(
                    self.decision_engine.memory_agent.memory_store
                )

                total_patterns = 0
                for (
                    memory_id,
                    memory_data,
                ) in self.decision_engine.memory_agent.memory_store.items():
                    patterns = memory_data.get("patterns", [])
                    total_patterns += len(patterns)
                insights["patterns_count"] = total_patterns

            try:
                import psutil

                insights["cpu"] = psutil.cpu_percent()
                insights["memory"] = (
                    psutil.virtual_memory().used // (1024 * 1024)
                )
            except ImportError:
                pass

        except Exception as e:
            print(f"Error getting agent insights: {e}")

        return insights

    def draw_detections_on_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw colored bounding boxes and labels on frame"""
        annotated_frame = frame.copy()

        weapon_detections = []
        person_detections = []

        for detection in detections:
            class_name = (
                detection.get("meta", {}).get("class_name", "Unknown").upper()
            )
            if class_name in ["GUN", "KNIFE", "WEAPON"]:
                weapon_detections.append(detection)
            elif class_name == "PERSON":
                person_detections.append(detection)

        # Draw persons first
        for detection in person_detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            person_id = detection.get("id", 0)
            person_id, activity, person_state = self.get_person_state_info(
                detection
            )

            color = self.human_tracker.get_id_color(person_id)

            is_occluded = detection.get("meta", {}).get(
                "is_occluded", False
            )
            keypoints = detection.get("meta", {}).get("keypoints", [])

            label_text = (
                f"ID: {person_id} | State: {person_state} | Pose: {activity}"
            )

            if is_occluded:
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    start_x = min(i, x2)
                    end_x = min(i + dash_length, x2)
                    cv2.line(
                        annotated_frame,
                        (start_x, y1),
                        (end_x, y1),
                        color,
                        3,
                    )
                    cv2.line(
                        annotated_frame,
                        (start_x, y2),
                        (end_x, y2),
                        color,
                        3,
                    )

                for i in range(y1, y2, dash_length * 2):
                    start_y = min(i, y2)
                    end_y = min(i + dash_length, y2)
                    cv2.line(
                        annotated_frame,
                        (x1, start_y),
                        (x1, end_y),
                        color,
                        3,
                    )
                    cv2.line(
                        annotated_frame,
                        (x2, start_y),
                        (x2, end_y),
                        color,
                        3,
                    )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.drawMarker(
                    annotated_frame,
                    (center_x, center_y),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    10,
                    3,
                )
                label_color = (0, 0, 255)
            else:
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2), color, 2
                )
                label_color = color

            label_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )[0]

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                label_color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            if len(keypoints) >= 17:
                try:
                    from detection.human_tracker import draw_pose_landmarks

                    annotated_frame = draw_pose_landmarks(
                        annotated_frame, keypoints, color, 2
                    )
                except ImportError:
                    pass

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if not is_occluded:
                cv2.circle(
                    annotated_frame, (center_x, center_y), 5, color, -1
                )
                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    7,
                    (255, 255, 255),
                    1,
                )

        # Draw weapons on top
        for detection in weapon_detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            weapon_class = (
                detection.get("meta", {}).get("class_name", "Unknown").upper()
            )
            weapon_type = detection.get("meta", {}).get("weapon_type", "Unknown")
            confidence = (
                detection.get("meta", {}).get("raw_confidence", 0) * 100
            )
            weapon_id = detection.get("id", 0)

            color = (0, 0, 255)
            label = weapon_class

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            label_text = f"{label} {confidence:.0f}%"
            label_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 30),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"ID:{weapon_id}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return annotated_frame

    def create_four_section_display(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        fire_smoke_result: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Create 4-section display layout"""
        height, width = frame.shape[:2]
        full_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        full_screen.fill(10)

        # Section 1: Original Feed with Detections
        annotated_frame = self.draw_detections_on_frame(frame, detections)
        if fire_smoke_result and (
            fire_smoke_result["fire_detected"]
            or fire_smoke_result["smoke_detected"]
        ):
            annotated_frame = (
                self.fire_smoke_detector.draw_fire_smoke_on_frame(
                    annotated_frame, fire_smoke_result
                )
            )
        original_section = cv2.resize(annotated_frame, (800, 600))
        full_screen[60:660, 0:800] = original_section

        # Section 2: Bird's Eye View
        birds_eye = self.create_birds_eye_view(frame, detections)
        birds_eye_small = cv2.resize(birds_eye, (240, 180))
        full_screen[60:240, 800:1040] = birds_eye_small

        # Section 3: Enhanced Heatmap
        heatmap = self.create_enhanced_heatmap(
            frame, self.detection_history
        )
        heatmap_small = cv2.resize(heatmap, (240, 180))
        full_screen[240:420, 800:1040] = heatmap_small

        # Section 4: System Analytics
        analytics = self.create_vertical_analytics_panel(frame, detections)
        full_screen[60:600, 1040:1280] = analytics

        # Section borders
        cv2.rectangle(full_screen, (0, 60), (800, 660), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (800, 60), (1040, 240), (0, 255, 0), 2)
        cv2.rectangle(
            full_screen, (800, 240), (1040, 420), (0, 255, 0), 2
        )
        cv2.rectangle(
            full_screen, (1040, 60), (1280, 600), (0, 255, 0), 2
        )

        # Section labels
        cv2.putText(
            full_screen,
            "LIVE FEED",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            full_screen,
            "BIRD'S EYE",
            (810, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            full_screen,
            "HEATMAP",
            (860, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            full_screen,
            "ANALYTICS",
            (1100, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Main header
        cv2.rectangle(
            full_screen, (0, 0), (1280, 60), (20, 20, 20), -1
        )
        cv2.putText(
            full_screen,
            "🎯 INTELLIGENT WEAPON DETECTION SYSTEM | AI-Powered Security Monitoring",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            full_screen,
            timestamp,
            (1050, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        # Bottom status bar
        cv2.rectangle(
            full_screen, (0, 660), (1280, 720), (20, 20, 20), -1
        )
        cv2.rectangle(
            full_screen, (0, 660), (1280, 720), (0, 255, 0), 1
        )

        current_state = getattr(
            self.decision_engine.state_agent, "state_transition", None
        )
        system_state = (
            current_state.current_state.value
            if current_state
            else "UNKNOWN"
        )

        # ✅ Show callback count in status bar
        cb_count = len(self._detection_callbacks)
        rt_count = len(self.recent_detections)

        status_text = (
            f"State: {system_state.upper()} | "
            f"Total: {self.stats['total_detections']} | "
            f"Threat: {self.stats['threat_detections']} | "
            f"FPS: {self.fps:.1f} | "
            f"Callbacks: {cb_count} | "
            f"RT Alerts: {rt_count}"
        )
        cv2.putText(
            full_screen,
            status_text,
            (20, 685),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence [W] Reset Recording"
        cv2.putText(
            full_screen,
            controls,
            (20, 705),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        return full_screen

    def generate_detection_alerts(
        self,
        detections: List[Dict[str, Any]],
        fire_smoke_result: Dict[str, Any],
        results: List[Dict[str, Any]],
    ):
        """Generate JSON alerts for all detections and store in Firebase"""
        current_alerts = []

        # Weapon detection alerts
        for detection in detections:
            if detection.get("meta", {}).get("class_name") in [
                "GUN",
                "KNIFE",
                "GRENADE",
                "EXPLOSION",
            ]:
                alert = self.alert_system.create_weapon_alert(
                    detection, self.frame_count
                )
                current_alerts.append(alert)
                self.alert_system.print_alert_json(
                    alert, "🚨 WEAPON DETECTED - BEEP BEEP BEEP 🚨"
                )

                alert_data = self.alert_system.alert_to_json(alert)
                parsed_alert = json.loads(alert_data)
                self.firebase_storage.store_alert(parsed_alert)

        # Violence detection alerts
        for detection in detections:
            if detection.get("violence_detected", False):
                alert = self.alert_system.create_violence_alert(
                    detection, self.frame_count
                )
                current_alerts.append(alert)
                self.alert_system.print_alert_json(
                    alert, "🥊 VIOLENCE DETECTED - BEEP BEEP BEEP 🚨"
                )

                alert_data = self.alert_system.alert_to_json(alert)
                parsed_alert = json.loads(alert_data)
                self.firebase_storage.store_alert(parsed_alert)

        # Fire detection alerts
        if fire_smoke_result.get("fire_detected", False):
            fire_alerts = self.alert_system.create_fire_alert(
                fire_smoke_result, self.frame_count
            )
            current_alerts.extend(fire_alerts)
            for alert in fire_alerts:
                self.alert_system.print_alert_json(
                    alert,
                    "🔥 FIRE DETECTED - EMERGENCY BEEP BEEP BEEP 🚨",
                )
                alert_data = self.alert_system.alert_to_json(alert)
                parsed_alert = json.loads(alert_data)
                self.firebase_storage.store_alert(parsed_alert)

        # Smoke detection alerts
        if fire_smoke_result.get("smoke_detected", False):
            smoke_alerts = self.alert_system.create_smoke_alert(
                fire_smoke_result, self.frame_count
            )
            current_alerts.extend(smoke_alerts)
            for alert in smoke_alerts:
                self.alert_system.print_alert_json(
                    alert,
                    "💨 SMOKE DETECTED - EMERGENCY BEEP BEEP BEEP 🚨",
                )
                alert_data = self.alert_system.alert_to_json(alert)
                parsed_alert = json.loads(alert_data)
                self.firebase_storage.store_alert(parsed_alert)

        # Suspicious pose alerts
        for detection in detections:
            if (
                detection.get("meta", {}).get("activity_type") == "aiming"
            ):
                alert = self.alert_system.create_pose_alert(
                    detection, self.frame_count, "AIMING"
                )
                current_alerts.append(alert)
                self.alert_system.print_alert_json(
                    alert,
                    "🎯 SUSPICIOUS AIMING DETECTED - BEEP BEEP 🚨",
                )
                alert_data = self.alert_system.alert_to_json(alert)
                parsed_alert = json.loads(alert_data)
                self.firebase_storage.store_alert(parsed_alert)

        # Store summary in Firebase
        if current_alerts:
            self.alert_system.print_summary_json(current_alerts)

            summary_data = self.alert_system.create_alert_summary(
                current_alerts
            )
            self.firebase_storage.store_alert_summary(summary_data)

            system_status = {
                "system_state": "EMERGENCY",
                "active_alerts": len(current_alerts),
                "camera_status": "ACTIVE",
                "detection_count": len(detections),
                "frame_count": self.frame_count,
                "timestamp": datetime.now().isoformat(),
            }
            self.firebase_storage.update_system_status(system_status)

            for alert in current_alerts:
                alert_id = alert.alert_id
                detection_type = alert.detection_type
                evidence_file = self.get_recent_evidence_file(
                    detection_type
                )
                if evidence_file and os.path.exists(evidence_file):
                    print(
                        f"📹 Storing evidence for {detection_type} "
                        f"alert: {alert_id}"
                    )
                    evidence_url = (
                        self.firebase_storage.store_evidence_file(
                            evidence_file, alert_id, detection_type
                        )
                    )
                    if evidence_url:
                        print(f"✓ Evidence stored: {evidence_url}")
                    else:
                        print(
                            f"❌ Failed to store evidence for "
                            f"{detection_type}"
                        )

    def get_recent_evidence_file(
        self, detection_type: str
    ) -> Optional[str]:
        """Get most recent evidence file for detection type"""
        try:
            evidence_dir = "evidence/videos"

            type_prefixes = {
                "WEAPON": "weapon_detection",
                "FIRE": "fire_detection",
                "SMOKE": "smoke_detection",
                "VIOLENCE": "violence_detection",
                "SUSPICIOUS_POSE": "suspicious_detection",
            }

            prefix = type_prefixes.get(detection_type, "detection")

            if not os.path.exists(evidence_dir):
                return None

            matching_files = []
            for file in os.listdir(evidence_dir):
                if file.startswith(prefix) and file.endswith(".mp4"):
                    file_path = os.path.join(evidence_dir, file)
                    mod_time = os.path.getmtime(file_path)
                    matching_files.append((mod_time, file_path))

            matching_files.sort(reverse=True)

            if matching_files:
                return matching_files[0][1]

            return None

        except Exception as e:
            print(f"❌ Error getting evidence file: {e}")
            return None

    def run(self):
        """Main detection loop"""
        if not self.start_camera():
            return

        print("\n=== INTEGRATED GUN DETECTION SYSTEM ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset statistics")
        print("Press 'e' to view evidence folder")
        print("Press 'w' to reset evidence session")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read error")
                    break

                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                self.frame_count += 1

                # ✅ Update FPS
                self._update_fps()

                # Detect objects
                detections, fire_smoke_result = self.detect_objects(frame)

                # Fire/smoke emergency
                if (
                    fire_smoke_result["fire_detected"]
                    or fire_smoke_result["smoke_detected"]
                ):
                    emergency_type = (
                        "FIRE"
                        if fire_smoke_result["fire_detected"]
                        else "SMOKE"
                    )
                    print(
                        f"🚨 EMERGENCY STATE TRIGGERED: "
                        f"{emergency_type} DETECTED!"
                    )

                    if hasattr(
                        self.decision_engine.state_agent,
                        "force_emergency_state",
                    ):
                        self.decision_engine.state_agent.force_emergency_state(
                            emergency_type
                        )

                    if fire_smoke_result["fire_detected"]:
                        self.evidence_agent.start_recording(
                            f"fire_detection_{self.frame_count}"
                        )
                    if fire_smoke_result["smoke_detected"]:
                        self.evidence_agent.start_recording(
                            f"smoke_detection_{self.frame_count}"
                        )

                # Process detections
                results = self.process_detections(detections, frame)

                # Generate alerts
                self.generate_detection_alerts(
                    detections, fire_smoke_result, results
                )

                # Create display
                display_frame = self.create_four_section_display(
                    frame, detections, results, fire_smoke_result
                )

                cv2.imshow(
                    "🎯 Intelligent Weapon Detection System",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.save_manual_frame(display_frame)
                elif key == ord("r"):
                    self.reset_statistics()
                elif key == ord("e"):
                    self.open_evidence_folder()
                elif key == ord("w"):
                    self.reset_evidence_session()

        except KeyboardInterrupt:
            print("\nDetection stopped by user")

        finally:
            self.cleanup()

    def save_manual_frame(self, frame: np.ndarray):
        """Manually save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_capture_{timestamp}.jpg"
        filepath = f"{self.evidence_folder}/{filename}"
        cv2.imwrite(filepath, frame)
        print(f"✓ Manual frame saved: {filename}")

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            "total_detections": 0,
            "threat_detections": 0,
            "alerts_triggered": 0,
            "evidence_saved": 0,
            "hands_up_detections": 0,
            "violence_detections": 0,
            "activity_detections": {},
        }
        self.recent_detections = []
        self.last_detections = []
        self.pose_detector.clear_poses()
        self.violence_detector.clear_fights()
        print("✓ Statistics reset")

    def open_evidence_folder(self):
        """Open evidence folder"""
        import subprocess

        try:
            if platform.system() == "Windows":
                os.startfile(self.evidence_folder)
            else:
                subprocess.run(["xdg-open", self.evidence_folder])
            print(f"✓ Evidence folder opened: {self.evidence_folder}")
        except Exception as e:
            print(f"Could not open evidence folder: {e}")

    def reset_evidence_session(self):
        """Reset evidence recording session"""
        if hasattr(self, "evidence_agent"):
            self.evidence_agent.reset_session()
            print(
                "✓ Evidence recording session reset - "
                "Ready for next detection"
            )
        else:
            print("❌ Evidence agent not available")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "evidence_agent"):
            self.evidence_agent.force_stop_recording()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")

        print("\n=== FINAL STATISTICS ===")
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # ✅ Print callback statistics
        print(f"\n=== CALLBACK STATISTICS ===")
        print(
            f"Registered Callbacks: {len(self._detection_callbacks)}"
        )
        print(
            f"Total Realtime Alerts Sent: {len(self.recent_detections)}"
        )

        if hasattr(self, "evidence_agent"):
            evidence_status = self.evidence_agent.get_status()
            print(f"\n=== EVIDENCE AGENT STATUS ===")
            print(
                f"Buffered Frames: {evidence_status['buffered_frames']}"
            )
            print(
                f"Final Recording: "
                f"{evidence_status['current_file'] or 'None'}"
            )
            print(f"Total Recordings: Check evidence/videos/ folder")


def main():
    """Main entry point"""
    print("=" * 60)
    print("INTEGRATED GUN DETECTION SYSTEM WITH ENHANCED UI")
    print("=" * 60)

    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} model not found!")
        print(f"Please ensure {model_path} is available")
        return

    try:
        system = IntegratedGunDetectionSystem(model_path=model_path)
        system.run()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()