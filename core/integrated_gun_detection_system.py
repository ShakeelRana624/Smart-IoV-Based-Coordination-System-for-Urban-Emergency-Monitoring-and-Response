"""
Complete Integrated Gun Detection System

Combines YOLO gun detection with agent-based decision making
for real-time threat assessment and response.
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
# from fight_detection import FightDetector  # Temporarily commented due to TensorFlow dependency issues
from agents.agent_based_decision_engine import AgentBasedDecisionEngine, AgentState

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
        
        # Initialize fight detector (temporarily disabled)
        # self.fight_detector = FightDetector()
        # print("✓ Fight detector initialized")
        self.fight_detector = None  # Temporarily disabled
        
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
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "threat_detections": 0,
            "alerts_triggered": 0,
            "evidence_saved": 0,
            "hands_up_detections": 0,
            "fight_detections": 0,
            "activity_detections": {}  # Track activity statistics
        }
    
    def init_evidence_storage(self):
        """Initialize evidence storage folder and database"""
        import os
        os.makedirs(self.evidence_folder, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.evidence_folder, "detections.db")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
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
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Evidence storage initialized: {self.evidence_folder}")
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_index}")
        
        # Set camera properties for color feed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Force RGB conversion
        # Try to set color format (may not work on all cameras)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        print(f"✓ Camera {self.camera_index} started")
        return True
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using YOLO model"""
        # Convert grayscale to color if needed
        if len(frame.shape) == 2:
            # Frame is grayscale, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Frame is single channel, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Frame is already 3-channel, ensure it's BGR
            pass
        
        # Detect weapons
        results = self.model(frame, stream=True, conf=0.5)
        weapon_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = self.convert_yolo_to_detection(box, frame)
                if detection:
                    weapon_detections.append(detection)
        
        # Detect humans
        human_detections = self.human_tracker.detect_humans(frame)
        
        # Combine detections
        all_detections = weapon_detections + human_detections
        
        return all_detections
    
    def convert_yolo_to_detection(self, box, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Convert YOLO detection to agent format"""
        try:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Class name mapping for better weapon classification
            original_class = self.model.names[cls].upper()
            
            # Enhanced weapon classification
            if "GUN" in original_class or "PISTOL" in original_class or "RIFLE" in original_class or "WEAPON" in original_class:
                class_name = "GUN"
                weapon_type = "Firearm"
            elif "KNIFE" in original_class or "BLADE" in original_class or "SHARP" in original_class:
                class_name = "KNIFE"
                weapon_type = "Blade Weapon"
            else:
                class_name = original_class
                weapon_type = "Unknown Weapon"
            
            # Create or update track
            person_id = self.update_track(x1, y1, w, h, class_name, conf)
            
            # Build detection dict
            detection = {
                "id": person_id,
                "bbox": [x1, y1, w, h],
                "person_conf": conf if class_name.lower() == "gun" else 0.0,
                "gun_conf": conf if class_name.lower() == "gun" else 0.0,
                "knife_conf": conf if class_name.lower() == "knife" else 0.0,
                "fight_conf": 0.0,  # Would need separate model
                "meta": {
                    "class_name": class_name,
                    "weapon_type": weapon_type,
                    "raw_confidence": conf,
                    "frame": self.frame_count,
                    "camera": self.camera_index
                },
                "timestamp": time.time(),
                "frame": frame.copy()
            }
            
            return detection
            
        except Exception as e:
            print(f"Error converting detection: {e}")
            return None
    
    def update_track(self, x: int, y: int, w: int, h: int, 
                   class_name: str, conf: float) -> int:
        """Update or create person tracks"""
        center_x, center_y = x + w//2, y + h//2
        
        # Find existing track
        best_match_id = None
        best_distance = float('inf')
        
        for track_id, track_info in self.active_tracks.items():
            track_center = track_info["center"]
            distance = np.sqrt((center_x - track_center[0])**2 + 
                             (center_y - track_center[1])**2)
            
            if distance < 100 and distance < best_distance:
                best_distance = distance
                best_match_id = track_id
        
        if best_match_id is None:
            # Create new track
            self.person_id_counter += 1
            best_match_id = self.person_id_counter
        
        # Update track
        self.active_tracks[best_match_id] = {
            "center": (center_x, center_y),
            "bbox": [x, y, w, h],
            "last_seen": self.frame_count,
            "class_name": class_name,
            "confidence": conf
        }
        
        # Clean old tracks
        self.clean_old_tracks()
        
        return best_match_id
    
    def clean_old_tracks(self):
        """Remove tracks not seen recently"""
        current_frame = self.frame_count
        stale_ids = []
        
        for track_id, track_info in self.active_tracks.items():
            if current_frame - track_info["last_seen"] > 30:
                stale_ids.append(track_id)
        
        for stale_id in stale_ids:
            del self.active_tracks[stale_id]
    
    def process_detections(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process detections through agent-based decision engine"""
        results = []
        
        # Store detection history for people counting
        if not hasattr(self, 'detection_history'):
            self.detection_history = []
        self.detection_history.append(detections)
        self.detection_history = self.detection_history[-100:]  # Keep last 100 frames
        
        # Add frame to evidence buffer for all frames (not just weapon detections)
        self.evidence_agent.add_frame_to_buffer(frame, time.time())
        
        # Detect poses for all persons in frame
        person_detections = [d for d in detections if d.get("class_name") == "PERSON"]
        if person_detections:
            # Detect poses
            pose_results = self.pose_detector.detect_poses_in_frame(frame, person_detections)
            
            # Update hands-up statistics
            hands_up_count = self.pose_detector.get_hands_up_count()
            if hands_up_count > self.stats.get("hands_up_detections", 0):
                self.stats["hands_up_detections"] = hands_up_count
                
                # Print hands-up detection info
                hands_up_ids = self.pose_detector.get_hands_up_person_ids()
                print(f"🙋 HANDS-UP DETECTED: {hands_up_count} persons with hands up (IDs: {hands_up_ids})")
            
            # Detect fights
            fight_results = self.fight_detector.detect_fights_in_frame(frame, person_detections)
            
            # Update fight statistics
            fight_count = self.fight_detector.get_fight_count()
            if fight_count > self.stats.get("fight_detections", 0):
                self.stats["fight_detections"] = fight_count
                
                # Print fight detection info
                fighting_ids = self.fight_detector.get_fighting_person_ids()
                print(f"🥊 FIGHT DETECTED: {fight_count} persons fighting (IDs: {fighting_ids})")
        
        for detection in detections:
            # Add pose information to detection if available
            person_id = detection.get("id")
            if person_id and person_id in self.pose_detector.detected_poses:
                pose_info = self.pose_detector.detected_poses[person_id]
                detection["pose_type"] = pose_info.get("pose_type", "NORMAL")
                detection["pose_confidence"] = pose_info.get("confidence", 0.0)
                detection["pose_keypoints"] = pose_info.get("keypoints", [])
            
            # Add fight information to detection if available (temporarily disabled)
            if self.fight_detector is not None and person_id and person_id in self.fight_detector.detected_fights:
                fight_info = self.fight_detector.detected_fights[person_id]
                detection["fight_detected"] = fight_info.get("fight_detected", False)
                detection["fight_confidence"] = fight_info.get("confidence", 0.0)
            
            # Process through agent engine
            result = self.decision_engine.process(detection)
            results.append(result)
            
            # Update statistics
            self.stats["total_detections"] += 1
            if result["threat_score"] > 1.0:  # Lowered threshold for testing
                self.stats["threat_detections"] += 1
            
            # Save evidence if needed (traditional image evidence)
            if "SAVE_EVIDENCE" in result["action"]:
                self.save_evidence(detection, result, frame)
            
            # Trigger alerts if needed
            if "LOCAL_ALARM" in result["action"]:
                self.trigger_alert(result)
            
            # Save to database
            self.save_to_database(detection, result, frame)
        
        return results
    
    def save_evidence(self, detection: Dict[str, Any], result: Dict[str, Any], frame: np.ndarray):
        """Save evidence frame"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_{detection['id']}_{timestamp}.jpg"
            filepath = f"{self.evidence_folder}/{filename}"
            
            # Draw annotations on frame
            annotated_frame = self.draw_annotations(frame, detection, result)
            cv2.imwrite(filepath, annotated_frame)
            
            self.stats["evidence_saved"] += 1
            print(f"✓ Evidence saved: {filename}")
            
        except Exception as e:
            print(f"Error saving evidence: {e}")
    
    def trigger_alert(self, result: Dict[str, Any]):
        """Trigger alert system"""
        current_time = time.time()
        
        # Prevent alert spam (minimum 2 seconds between alerts)
        if current_time - self.last_alert_time < 2:
            return
        
        self.last_alert_time = current_time
        self.stats["alerts_triggered"] += 1
        
        # Play alert sound
        self.play_alert_sound(result["state"])
        
        # Print alert
        print(f"🚨 ALERT TRIGGERED: {result['state']} (Score: {result['threat_score']:.2f})")
    
    def play_alert_sound(self, threat_level: str):
        """Play alert sound based on threat level"""
        try:
            if platform.system() == "Windows":
                import winsound
                if threat_level in ["CRITICAL", "VIOLENT"]:
                    winsound.Beep(1500, 500)  # High frequency, longer
                elif threat_level in ["HIGH", "ARMED"]:
                    winsound.Beep(1000, 300)  # Medium frequency
                else:
                    winsound.Beep(800, 200)   # Lower frequency
            else:
                print('\a')  # System bell for Unix
        except Exception as e:
            print(f"Alert sound failed: {e}")
    
    def save_to_database(self, detection: Dict[str, Any], result: Dict[str, Any], frame: np.ndarray):
        """Save detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Compress frame for storage
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_data = buffer.tobytes()
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, detection_id, bbox, confidence, threat_level, threat_score, actions, evidence_path, frame_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection["timestamp"],
                detection["id"],
                json.dumps(detection["bbox"]),
                detection.get("person_conf", 0),
                result["state"],
                result["threat_score"],
                result["action"],
                f"evidence_{detection['id']}.jpg",
                frame_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
    
    def draw_annotations(self, frame: np.ndarray, detection: Dict[str, Any], result: Dict[str, Any]) -> np.ndarray:
        """Draw professional annotations on frame"""
        annotated = frame.copy()
        
        # Bounding box with enhanced styling
        bbox = detection["bbox"]
        x, y, w, h = bbox
        
        # Get color based on system state
        system_state = result.get("system_state", "normal").upper()
        color = self.get_system_state_color(system_state)
        
        # Draw thick bounding box with glow effect
        for i in range(3):
            alpha = 255 - (i * 80)
            cv2.rectangle(annotated, (x-i, y-i), (x + w + i, y + h + i), 
                         (color[0], color[1], color[2]), 1)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Enhanced label with background
        label = f"ID:{detection['id']} {system_state}"
        score_text = f"Score:{result['threat_score']:.1f}"
        
        # Calculate label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        bg_height = 60
        cv2.rectangle(annotated, (x, y - bg_height), (x + label_size[0] + 20, y), color, -1)
        cv2.rectangle(annotated, (x, y - bg_height), (x + label_size[0] + 20, y), (255, 255, 255), 1)
        
        # Text with shadow effect
        cv2.putText(annotated, label, (x + 10, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, score_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add fight information if available
        fight_detected = detection.get("fight_detected", False)
        fight_confidence = detection.get("fight_confidence", 0.0)
        
        if fight_detected:
            # Draw fight detection with red styling
            fight_color = (0, 0, 255)  # Red for fight
            fight_label = f"🥊 FIGHT ID:{detection['id']}"
            fight_conf_text = f"Fight:{fight_confidence:.2f}"
            
            # Draw fight indicator above all other elements
            fight_label_size = cv2.getTextSize(fight_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x, y - 105), (x + fight_label_size[0] + 10, y - 80), fight_color, -1)
            cv2.putText(annotated, fight_label, (x + 5, y - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, fight_conf_text, (x + 5, y - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fight_color, 1)
            
            # Draw warning text
            cv2.putText(annotated, "⚠️ VIOLENCE DETECTED", (x, y - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fight_color, 2)
            
            # Draw fight skeleton on frame
            fight_info = {
                "person_id": detection['id'],
                "fight_detected": fight_detected,
                "confidence": fight_confidence,
                "bbox": bbox
            }
            annotated = self.fight_detector.draw_fight_on_frame(annotated, fight_info, fight_color)
        
        # Add pose information if available
        pose_type = detection.get("pose_type", "NORMAL")
        pose_confidence = detection.get("pose_confidence", 0.0)
        
        if pose_type == "HANDS_UP":
            # Draw hands-up indicator with special styling
            pose_color = (0, 255, 255)  # Yellow for hands-up
            pose_label = f"🙋 HANDS_UP"
            pose_conf_text = f"Pose:{pose_confidence:.2f}"
            
            # Draw pose indicator above bounding box (but below fight indicator)
            pose_label_size = cv2.getTextSize(pose_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x, y - 85), (x + pose_label_size[0] + 10, y - 60), pose_color, -1)
            cv2.putText(annotated, pose_label, (x + 5, y - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(annotated, pose_conf_text, (x + 5, y - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pose_color, 1)
            
            # Draw pose skeleton on frame
            pose_keypoints = detection.get("pose_keypoints", [])
            if pose_keypoints:
                pose_info = {
                    "person_id": detection['id'],
                    "pose_type": pose_type,
                    "confidence": pose_confidence,
                    "keypoints": pose_keypoints,
                    "bbox": bbox
                }
                annotated = self.pose_detector.draw_pose_on_frame(annotated, pose_info, pose_color)
        
        # Add weapon confidence indicators
        confidences = []
        if detection.get("gun_conf", 0) > 0.1:
            confidences.append(f"🔫 Gun:{detection['gun_conf']:.2f}")
        if detection.get("knife_conf", 0) > 0.1:
            confidences.append(f"🔪 Knife:{detection['knife_conf']:.2f}")
        if detection.get("explosion_conf", 0) > 0.1:
            confidences.append(f"💥 Explosion:{detection['explosion_conf']:.2f}")
        if detection.get("grenade_conf", 0) > 0.1:
            confidences.append(f"🧨 Grenade:{detection['grenade_conf']:.2f}")
        
        # Draw confidence indicators
        if confidences:
            conf_text = " | ".join(confidences[:3])  # Limit to 3 items
            cv2.putText(annotated, conf_text, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add state change indicator with animation effect
        if result.get("state_changed", False):
            cv2.putText(annotated, "⚡ STATE CHANGED!", (x, y + h + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add emergency indicator if active
        emergency_response = result.get("emergency_response")
        if emergency_response and not emergency_response.get("emergency_deactivated"):
            threat_type = emergency_response.get("threat_type", "UNKNOWN")
            cv2.putText(annotated, f"🚨 EMERGENCY: {threat_type}", 
                       (annotated.shape[1] // 2 - 150, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        return annotated
    
    def get_system_state_color(self, state: str) -> tuple:
        """Get color based on system state"""
        colors = {
            "NORMAL": (0, 255, 0),      # Green
            "SUSPICIOUS": (0, 255, 255), # Yellow
            "THREAT_DETECTION": (0, 165, 255), # Orange
            "EMERGENCY": (0, 0, 255),    # Red
            # Legacy compatibility
            "MINIMAL": (0, 255, 0),
            "LOW": (255, 255, 0),
            "MEDIUM": (255, 165, 0),
            "HIGH": (255, 0, 0),
            "CRITICAL": (255, 0, 0),
            "VIOLENT": (0, 0, 255),
            "ARMED": (255, 165, 0)
        }
        return colors.get(state, (255, 255, 255))
    
    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw professional statistics on frame"""
        stats_frame = frame.copy()
        
        # Create professional overlay panels
        height, width = stats_frame.shape[:2]
        
        # Top header bar
        cv2.rectangle(stats_frame, (0, 0), (width, 60), (20, 20, 20), -1)
        cv2.rectangle(stats_frame, (0, 0), (width, 60), (0, 255, 0), 2)
        
        # System title
        cv2.putText(stats_frame, "INTELLIGENT WEAPON DETECTION SYSTEM", (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(stats_frame, "AI-Powered Security Monitoring", (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Right side status panel (smaller for 4-section layout)
        panel_width = 200
        cv2.rectangle(stats_frame, (width - panel_width, 60), (width, height - 40), (30, 30, 30), -1)
        cv2.rectangle(stats_frame, (width - panel_width, 60), (width, height - 40), (0, 255, 0), 2)
        
        # Get current system state
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "UNKNOWN"
        state_color = self.get_system_state_color(system_state.upper())
        
        # Status header
        cv2.putText(stats_frame, "STATUS", (width - panel_width + 10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw state indicator with color
        cv2.rectangle(stats_frame, (width - panel_width + 10, 90), 
                     (width - panel_width + 25, 105), state_color, -1)
        cv2.putText(stats_frame, f"{system_state[:4].upper()}", (width - panel_width + 30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 2)
        
        # Compact statistics
        stats_text = [
            f"Total: {self.stats['total_detections']}",
            f"Threat: {self.stats['threat_detections']}",
            f"Alerts: {self.stats['alerts_triggered']}",
            f"Evidence: {self.stats['evidence_saved']}",
            "━━━━━━━━━━━━",
            f"FPS: {30:.1f}",
        ]
        
        y_offset = 125
        for text in stats_text:
            color = (255, 255, 255) if text.startswith(("Total", "Threat", "Alerts", "Evidence")) else (100, 100, 100)
            if text.startswith("FPS:"):
                color = (0, 255, 0)
            cv2.putText(stats_frame, text, (width - panel_width + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 18
        
        # Bottom control bar
        cv2.rectangle(stats_frame, (0, height - 40), (width, height), (20, 20, 20), -1)
        cv2.rectangle(stats_frame, (0, height - 40), (width, height), (0, 255, 0), 1)
        
        # Controls info
        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence [W] Reset Recording"
        cv2.putText(stats_frame, controls, (20, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(stats_frame, timestamp, (width - 100, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return stats_frame
    
    def create_birds_eye_view(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create bird's eye view of detection area"""
        # Create blank canvas for bird's eye view
        birds_eye = np.zeros((240, 320, 3), dtype=np.uint8)
        birds_eye.fill(20)  # Dark background
        
        # Draw grid
        for i in range(0, 320, 40):
            cv2.line(birds_eye, (i, 0), (i, 240), (40, 40, 40), 1)
        for i in range(0, 240, 40):
            cv2.line(birds_eye, (0, i), (320, i), (40, 40, 40), 1)
        
        # Draw detection area boundary
        cv2.rectangle(birds_eye, (10, 10), (310, 230), (0, 255, 0), 2)
        
        # Add title
        cv2.putText(birds_eye, "BIRD'S EYE VIEW", (80, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Plot detections as circles
        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                # Convert to bird's eye view coordinates
                bird_x = int((x + w/2) * 320 / frame.shape[1])
                bird_y = int((y + h/2) * 240 / frame.shape[0])
                
                # Get threat level
                threat_score = detection.get("threat_score", 0)
                if threat_score > 2.0:
                    color = (0, 0, 255)  # Red for high threat
                elif threat_score > 1.0:
                    color = (0, 165, 255)  # Orange for medium threat
                else:
                    color = (0, 255, 0)  # Green for low threat
                
                # Draw detection point
                cv2.circle(birds_eye, (bird_x, bird_y), 8, color, -1)
                cv2.circle(birds_eye, (bird_x, bird_y), 10, color, 2)
                
                # Add ID
                cv2.putText(birds_eye, str(detection.get("id", "?")), 
                           (bird_x - 5, bird_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return birds_eye
    
    def create_enhanced_heatmap(self, frame: np.ndarray, detections_history: List[Dict[str, Any]]) -> np.ndarray:
        """Create enhanced activity heatmap with better visualization"""
        heatmap = np.zeros((180, 240, 3), dtype=np.uint8)
        heatmap.fill(20)  # Dark background
        
        # Add title
        cv2.putText(heatmap, "ACTIVITY HEATMAP", (50, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Create enhanced heatmap based on detection history
        if detections_history:
            # Draw fine grid
            grid_size = 20
            for i in range(0, 240, grid_size):
                cv2.line(heatmap, (i, 30), (i, 180), (30, 30, 30), 1)
            for i in range(30, 180, grid_size):
                cv2.line(heatmap, (0, i), (240, i), (30, 30, 30), 1)
            
            # Calculate activity density with higher resolution
            activity_grid = np.zeros((8, 12), dtype=float)  # Higher resolution grid
            
            for detection_batch in detections_history[-100:]:  # Last 100 detection batches
                if isinstance(detection_batch, list):  # Check if it's a list of detections
                    for detection in detection_batch:
                        if isinstance(detection, dict):  # Check if it's a detection dictionary
                            bbox = detection.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                x, y, w, h = bbox[:4]
                                center_x = int((x + w/2) / frame.shape[1] * 12)
                                center_y = int((y + h/2) / frame.shape[0] * 8)
                                
                                if 0 <= center_x < 12 and 0 <= center_y < 8:
                                    # Add weighted contribution based on threat score
                                    threat_score = detection.get("threat_score", 0.5)
                                    activity_grid[center_y, center_x] += threat_score
            
            # Apply Gaussian smoothing for better visualization
            from scipy.ndimage import gaussian_filter
            try:
                activity_grid = gaussian_filter(activity_grid, sigma=1.0)
            except:
                pass  # Fallback if scipy not available
            
            # Draw enhanced heatmap with gradient
            max_activity = np.max(activity_grid) if np.max(activity_grid) > 0 else 1
            
            for y in range(8):
                for x in range(12):
                    intensity = activity_grid[y, x] / max_activity
                    
                    # Enhanced color gradient
                    if intensity < 0.2:
                        # Dark blue to blue
                        color = (0, 0, int(intensity * 5 * 255))
                    elif intensity < 0.4:
                        # Blue to cyan
                        ratio = (intensity - 0.2) / 0.2
                        color = (0, int(ratio * 255), 255)
                    elif intensity < 0.6:
                        # Cyan to green
                        ratio = (intensity - 0.4) / 0.2
                        color = (0, 255, int((1 - ratio) * 255))
                    elif intensity < 0.8:
                        # Green to yellow
                        ratio = (intensity - 0.6) / 0.2
                        color = (int(ratio * 255), 255, 0)
                    else:
                        # Yellow to red
                        ratio = (intensity - 0.8) / 0.2
                        color = (255, int((1 - ratio) * 255), 0)
                    
                    # Draw cell with gradient effect
                    cell_x = x * grid_size
                    cell_y = 30 + y * grid_size
                    
                    # Main cell
                    cv2.rectangle(heatmap, (cell_x, cell_y), 
                                 (cell_x + grid_size, cell_y + grid_size), color, -1)
                    
                    # Add subtle border
                    cv2.rectangle(heatmap, (cell_x, cell_y), 
                                 (cell_x + grid_size, cell_y + grid_size), 
                                 (color[0]//2, color[1]//2, color[2]//2), 1)
                    
                    # Add activity count for high activity areas
                    if activity_grid[y, x] > 0.5:
                        count = int(activity_grid[y, x])
                        cv2.putText(heatmap, str(count), 
                                   (cell_x + 5, cell_y + 12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add intensity scale
            scale_x = 220
            scale_height = 120
            for i in range(scale_height):
                intensity = i / scale_height
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
                
                cv2.line(heatmap, (scale_x, 30 + scale_height - i), 
                        (scale_x + 10, 30 + scale_height - i), color, 2)
            
            # Scale labels
            cv2.putText(heatmap, "HIGH", (scale_x - 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            cv2.putText(heatmap, "LOW", (scale_x - 5, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        return heatmap
    
    def get_activity_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get activity statistics from detections"""
        activity_stats = {
            "Sitting": 0,
            "Standing": 0,
            "Walking": 0,
            "Running": 0,
            "Lying": 0,
            "HandsUp": 0,
            "Aiming": 0,
            "Unknown": 0
        }
        
        for detection in detections:
            if detection.get("meta", {}).get("class_name") == "PERSON":
                activity = detection.get("meta", {}).get("activity", "Unknown")
                if activity in activity_stats:
                    activity_stats[activity] += 1
                else:
                    activity_stats["Unknown"] += 1
        
        return activity_stats
    
    def get_enhanced_system_state(self, detections: List[Dict[str, Any]]) -> str:
        """Enhanced state management based on weapon and pose combinations"""
        has_weapon = False
        has_aiming = False
        has_hands_up = False
        weapon_count = 0
        aiming_count = 0
        hands_up_count = 0
        
        # Check all detections for weapons and poses
        for detection in detections:
            class_name = detection.get("meta", {}).get("class_name", "").upper()
            activity = detection.get("meta", {}).get("activity", "Unknown")
            
            # Check for weapons
            if class_name in ["GUN", "KNIFE", "WEAPON"]:
                has_weapon = True
                weapon_count += 1
            
            # Check for suspicious poses
            if activity.upper() in ["AIMING", "HANDSUP"]:
                if activity.upper() == "AIMING":
                    has_aiming = True
                    aiming_count += 1
                elif activity.upper() == "HANDSUP":
                    has_hands_up = True
                    hands_up_count += 1
        
        # Enhanced state logic
        if has_weapon and (has_aiming or has_hands_up):
            # EMERGENCY: Weapon + Suspicious Pose
            print(f"🚨 EMERGENCY STATE: Weapon detected ({weapon_count}) + Suspicious pose (Aiming: {aiming_count}, HandsUp: {hands_up_count})")
            self.trigger_emergency_alert()
            return "EMERGENCY"
        elif has_weapon:
            # WEAPON DETECTED: Only weapon, no suspicious pose
            print(f"⚠️ WEAPON DETECTED: {weapon_count} weapons found")
            return "WEAPON_DETECTED"
        elif has_aiming or has_hands_up:
            # SUSPICIOUS: Only suspicious poses, no weapon
            print(f"🔍 SUSPICIOUS STATE: Aiming ({aiming_count}) or HandsUp ({hands_up_count}) detected")
            return "SUSPICIOUS"
        else:
            # NORMAL: No weapons, no suspicious poses
            return "NORMAL"
    
    def trigger_emergency_alert(self):
        """Trigger emergency alert with beep"""
        try:
            import winsound
            # Play emergency beep sound
            winsound.Beep(1000, 500)  # 1000Hz for 500ms
            print("🚨 EMERGENCY ALERT BEEP ACTIVATED!")
        except ImportError:
            print("🚨 EMERGENCY ALERT (Visual Only - winsound not available)")
        except Exception as e:
            print(f"🚨 EMERGENCY ALERT (Sound Error: {e})")
    
    def get_person_state_info(self, detection: Dict[str, Any]) -> tuple:
        """Get individual person's state information"""
        person_id = detection.get("id", 0)
        activity = detection.get("meta", {}).get("activity", "Unknown")
        class_name = detection.get("meta", {}).get("class_name", "").upper()
        
        # Determine person state
        person_state = "NORMAL"
        if class_name in ["GUN", "KNIFE", "WEAPON"]:
            person_state = "WEAPON_DETECTED"
        elif activity.upper() in ["AIMING", "HANDSUP"]:
            person_state = "SUSPICIOUS"
        
        return person_id, activity, person_state
    
    def create_vertical_analytics_panel(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        """Create vertical analytics panel for right side"""
        analytics = np.zeros((540, 240, 3), dtype=np.uint8)
        analytics.fill(20)  # Dark background
        
        # Add title
        cv2.putText(analytics, "SYSTEM ANALYTICS", (50, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Get current system state using enhanced logic
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = self.get_enhanced_system_state(detections) if detections else "NORMAL"
        
        # Analytics data
        hands_up_count = self.pose_detector.get_hands_up_count()
        hands_up_ids = self.pose_detector.get_hands_up_person_ids()
        
        # Fight detection (temporarily disabled)
        fight_count = 0
        fighting_ids = []
        
        # Get activity statistics from human tracker
        activity_stats = self.get_activity_statistics(detections)
        
        # Get current and total people count
        self.current_detections = detections  # Store current detections
        current_people = len([d for d in detections if d.get("meta", {}).get("class_name") == "PERSON"])
        all_person_ids = set()
        if hasattr(self, 'detection_history'):
            for detection_batch in self.detection_history[-100:]:  # Last 100 frames
                if isinstance(detection_batch, list):  # Check if it's a list of detections
                    for detection in detection_batch:
                        if isinstance(detection, dict) and detection.get("meta", {}).get("class_name") == "PERSON":
                            all_person_ids.add(detection.get("id", 0))
        total_unique_people = len(all_person_ids)
        
        analytics_data = [
            ("System State:", system_state.upper(), self.get_system_state_color(system_state.upper())),
            ("", "", (255, 255, 255)),
            ("📊 Statistics", "", (0, 255, 0)),
            ("Total:", str(self.stats['total_detections']), (255, 255, 255)),
            ("Threat:", str(self.stats['threat_detections']), (255, 165, 0)),
            ("Current People:", str(current_people), (0, 255, 0)),  # Added current people count
            ("Total Unique:", str(total_unique_people), (0, 255, 255)),  # Added total unique people
            ("Hands-Up:", str(hands_up_count), (0, 255, 255)),
            ("Fights:", str(fight_count), (0, 0, 255)),  # Added fight count
            ("Threat Rate:", f"{(self.stats['threat_detections']/max(1,self.stats['total_detections'])*100):.1f}%", (255, 165, 0)),
            ("Evidence:", str(len([f for f in os.listdir("evidence/videos") if f.endswith(".mp4")])), (0, 255, 255)),
            ("", "", (255, 255, 255)),
            ("🙋 Pose Status", "", (0, 255, 255)),
            (f"Active IDs:", str(hands_up_ids) if hands_up_ids else "None", (0, 255, 255)),
            ("", "", (255, 255, 255)),
            ("🥊 Fight Status", "", (0, 0, 255)),  # Added fight status section
            (f"Fighting IDs:", str(fighting_ids) if fighting_ids else "None", (0, 0, 255)),
            ("", "", (255, 255, 255)),
            ("🤸 Activity Status", "", (255, 165, 0)),  # Added activity status section
            (f"Sitting:", str(activity_stats.get("Sitting", 0)), (255, 165, 0)),
            (f"Standing:", str(activity_stats.get("Standing", 0)), (0, 255, 0)),
            (f"Walking:", str(activity_stats.get("Walking", 0)), (0, 255, 255)),
            (f"Running:", str(activity_stats.get("Running", 0)), (255, 0, 0)),
            (f"HandsUp:", str(activity_stats.get("HandsUp", 0)), (255, 255, 0)),
            (f"Aiming:", str(activity_stats.get("Aiming", 0)), (0, 0, 255)),
            ("", "", (255, 255, 255)),
            ("⚡ Performance", "", (0, 255, 0)),
            ("FPS:", "30.0", (0, 255, 0)),
            ("CPU:", "45%", (255, 255, 0)),
            ("Memory:", "512MB", (255, 165, 0)),
            ("", "", (255, 255, 255)),
            ("⏱️ Timeline", "", (0, 255, 255)),
            ("Uptime:", "02:15:30", (0, 255, 255)),
            ("Last Alert:", "None", (255, 255, 255)),
            ("", "", (255, 255, 255)),
            ("🔥 Activity", "", (255, 165, 0)),
            ("Last Hour:", str(len([d for d in getattr(self, 'detection_history', [])[-50:]])), (255, 165, 0)),
            ("Peak:", "12/min", (255, 0, 0)),
        ]
        
        y_offset = 50
        for label, value, color in analytics_data:
            if label:
                if label.startswith(("📊", "⚡", "⏱️", "🔥")):
                    # Category headers
                    cv2.putText(analytics, label, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Regular labels and values
                    cv2.putText(analytics, label, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    cv2.putText(analytics, value, (120, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 18
        
        # Add weapon details section
        cv2.putText(analytics, "WEAPONS DETECTED", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 20
        
        if detections is None:
            detections = []
        
        if detections:
            for i, detection in enumerate(detections[:5]):  # Show max 5 weapons
                weapon_type = detection.get("meta", {}).get("weapon_type", "Unknown").upper()
                weapon_class = detection.get("meta", {}).get("class_name", "Unknown").upper()
                weapon_id = detection.get("id", i+1)
                confidence = detection.get("meta", {}).get("raw_confidence", 0) * 100
                
                # Enhanced color coding - Red for all weapons
                if weapon_class == "GUN" or "FIREARM" in weapon_type:
                    color = (0, 0, 255)  # Red for guns
                    icon = "🔫"
                elif weapon_class == "KNIFE" or "BLADE" in weapon_type:
                    color = (0, 0, 255)  # Red for knives too
                    icon = "🔪"
                else:
                    color = (0, 0, 255)  # Red for all weapons
                    icon = "⚠️"
                
                # Display with weapon type and icon
                cv2.putText(analytics, f"{icon} ID:{weapon_id} {weapon_class}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                cv2.putText(analytics, f"{weapon_type}", (10, y_offset + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                cv2.putText(analytics, f"{confidence:.1f}%", (150, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += 25
        else:
            cv2.putText(analytics, "No weapons detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            y_offset += 15
        
        return analytics
    
    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw colored bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        # Separate weapons and persons
        weapon_detections = []
        person_detections = []
        
        for detection in detections:
            class_name = detection.get("meta", {}).get("class_name", "Unknown").upper()
            if class_name in ["GUN", "KNIFE", "WEAPON"]:
                weapon_detections.append(detection)
            elif class_name == "PERSON":
                person_detections.append(detection)
        
        # Draw persons first (green boxes - background layer)
        for detection in person_detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            confidence = detection.get("meta", {}).get("raw_confidence", 0) * 100
            person_id = detection.get("id", 0)
            
            # Get person state information
            person_id, activity, person_state = self.get_person_state_info(detection)
            
            # Get consistent color from human tracker
            color = self.human_tracker.get_id_color(person_id)
            
            # Check if track is occluded (from DeepSort)
            is_occluded = detection.get("meta", {}).get("is_occluded", False)
            time_since_update = detection.get("meta", {}).get("time_since_update", 0)
            keypoints = detection.get("meta", {}).get("keypoints", [])
            
            # Enhanced label with State, ID, and Pose
            label_lines = [
                f"ID: {person_id}",
                f"State: {person_state}",
                f"Pose: {activity}"
            ]
            
            # Draw bounding box with occlusion handling
            if is_occluded:
                # Draw dashed bounding box for occluded person
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    start_x = min(i, x2)
                    end_x = min(i + dash_length, x2)
                    cv2.line(annotated_frame, (start_x, y1), (end_x, y1), color, 3)
                    cv2.line(annotated_frame, (start_x, y2), (end_x, y2), color, 3)
                
                for i in range(y1, y2, dash_length * 2):
                    start_y = min(i, y2)
                    end_y = min(i + dash_length, y2)
                    cv2.line(annotated_frame, (x1, start_y), (x1, end_y), color, 3)
                    cv2.line(annotated_frame, (x2, start_y), (x2, end_y), color, 3)
                
                # Red dot indicator for occluded person
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.drawMarker(annotated_frame, (center_x, center_y), (0, 0, 255), 
                             cv2.MARKER_CROSS, 10, 3)
                
                # Occlusion label with state info
                label_color = (0, 0, 255)  # Red for occluded
            else:
                # Solid bounding box for visible person
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Normal label with state info
                label_color = color
            
            # Draw multi-line label background and text
            label_height = 25
            total_label_height = len(label_lines) * label_height + 5
            
            # Draw background for multi-line label
            cv2.rectangle(annotated_frame, (x1, y1 - total_label_height), 
                         (x1 + 150, y1), label_color, -1)
            
            # Draw each line of the label
            for i, line in enumerate(label_lines):
                line_y = y1 - total_label_height + (i + 1) * label_height - 5
                cv2.putText(annotated_frame, line, (x1 + 5, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw pose landmarks if available
            if len(keypoints) >= 17:
                # Import pose drawing functions from human_tracker
                from detection.human_tracker import draw_pose_landmarks, draw_activity_info
                annotated_frame = draw_pose_landmarks(annotated_frame, keypoints, color, 2)
                # Comment out draw_activity_info to avoid duplicate labels
                # annotated_frame = draw_activity_info(annotated_frame, person_id, activity, keypoints, (x1, y1, x2, y2))
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if is_occluded:
                # Draw X for occluded person
                cv2.drawMarker(annotated_frame, (center_x, center_y), label_color, 
                             cv2.MARKER_CROSS, 10, 3)
            else:
                # Draw filled circle for visible person
                cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
                cv2.circle(annotated_frame, (center_x, center_y), 7, (255, 255, 255), 1)
        
        # Draw weapons on top (red boxes - foreground layer)
        for detection in weapon_detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            weapon_class = detection.get("meta", {}).get("class_name", "Unknown").upper()
            weapon_type = detection.get("meta", {}).get("weapon_type", "Unknown").upper()
            confidence = detection.get("meta", {}).get("raw_confidence", 0) * 100
            weapon_id = detection.get("id", 0)
            
            # Color coding - Red for all weapons with clear labels
            if weapon_class == "GUN" or "FIREARM" in weapon_type:
                color = (0, 0, 255)  # Red for guns
                label = "GUN"
            elif weapon_class == "KNIFE" or "BLADE" in weapon_type:
                color = (0, 0, 255)  # Red for knives too
                label = "KNIFE"
            else:
                color = (0, 0, 255)  # Red for all weapons
                label = weapon_class
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)  # Thicker for weapons
            
            # Draw label background and text
            label_text = f"{label} {confidence:.0f}%"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text in white
            cv2.putText(annotated_frame, label_text, (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw ID
            cv2.putText(annotated_frame, f"ID:{weapon_id}", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame

    def create_four_section_display(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                                   results: List[Dict[str, Any]]) -> np.ndarray:
        """Create 4-section display layout with larger main feed"""
        # Get original frame dimensions
        height, width = frame.shape[:2]
        
        # Create full screen canvas (1280x720)
        full_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        full_screen.fill(10)  # Dark background
        
        # Section 1: Original Feed with Detections (Left Side - 800x600) - Larger
        annotated_frame = self.draw_detections_on_frame(frame, detections)
        original_section = cv2.resize(annotated_frame, (800, 600))
        full_screen[60:660, 0:800] = original_section
        
        # Section 2: Bird's Eye View (Top Right - 240x180) - Smaller
        birds_eye = self.create_birds_eye_view(frame, detections)
        birds_eye_small = cv2.resize(birds_eye, (240, 180))
        full_screen[60:240, 800:1040] = birds_eye_small
        
        # Section 3: Enhanced Heatmap (Bottom Right - 240x180) - Smaller but Enhanced
        heatmap = self.create_enhanced_heatmap(frame, self.detection_history)
        heatmap_small = cv2.resize(heatmap, (240, 180))
        full_screen[240:420, 800:1040] = heatmap_small
        
        # Section 4: System Analytics (Right Side - 240x540) - Vertical Panel
        analytics = self.create_vertical_analytics_panel(frame, detections)
        full_screen[60:600, 1040:1280] = analytics
        
        # Add section borders
        cv2.rectangle(full_screen, (0, 60), (800, 660), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (800, 60), (1040, 240), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (800, 240), (1040, 420), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (1040, 60), (1280, 600), (0, 255, 0), 2)
        
        # Add section labels
        cv2.putText(full_screen, "LIVE FEED", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(full_screen, "BIRD'S EYE", (810, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(full_screen, "HEATMAP", (860, 265), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(full_screen, "ANALYTICS", (1100, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add main header
        cv2.rectangle(full_screen, (0, 0), (1280, 60), (20, 20, 20), -1)
        cv2.putText(full_screen, "🎯 INTELLIGENT WEAPON DETECTION SYSTEM | AI-Powered Security Monitoring", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(full_screen, timestamp, (1050, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Add bottom status bar
        cv2.rectangle(full_screen, (0, 660), (1280, 720), (20, 20, 20), -1)
        cv2.rectangle(full_screen, (0, 660), (1280, 720), (0, 255, 0), 1)
        
        # Status info
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "UNKNOWN"
        state_color = self.get_system_state_color(system_state.upper())
        
        status_text = f"State: {system_state.upper()} | Total: {self.stats['total_detections']} | Threat: {self.stats['threat_detections']} | FPS: 30.0"
        cv2.putText(full_screen, status_text, (20, 685), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence"
        cv2.putText(full_screen, controls, (20, 705), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return full_screen
    
    def run(self):
        """Main detection loop"""
        if not self.start_camera():
            return
        
        print("\n=== INTEGRATED GUN DETECTION SYSTEM ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset statistics")
        print("Press 'e' to view evidence folder")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read error")
                    break
                
                # Convert frame to color if grayscale
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                self.frame_count += 1
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Process detections
                results = self.process_detections(detections, frame)
                
                # Create 4-section display
                display_frame = self.create_four_section_display(frame, detections, results)
                
                # Display frame with professional window title
                cv2.imshow("🎯 Intelligent Weapon Detection System | AI-Powered Security", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_manual_frame(display_frame)  # Save full 4-section display
                elif key == ord('r'):
                    self.reset_statistics()
                elif key == ord('e'):
                    self.open_evidence_folder()
                elif key == ord('w'):
                    self.reset_evidence_session()  # Reset evidence recording session
        
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
            "fight_detections": 0
        }
        # Clear pose detector poses
        self.pose_detector.clear_poses()
        # Clear fight detector fights
        self.fight_detector.clear_fights()
        print("✓ Statistics reset")
    
    def open_evidence_folder(self):
        """Open evidence folder"""
        import subprocess
        import os
        try:
            if platform.system() == "Windows":
                os.startfile(self.evidence_folder)
            else:
                subprocess.run(["xdg-open", self.evidence_folder])
            print(f"✓ Evidence folder opened: {self.evidence_folder}")
        except Exception as e:
            print(f"Could not open evidence folder: {e}")
    
    def reset_evidence_session(self):
        """Reset evidence recording session for new weapon detection"""
        if hasattr(self, 'evidence_agent'):
            self.evidence_agent.reset_session()
            print("✓ Evidence recording session reset - Ready for next weapon detection")
        else:
            print("❌ Evidence agent not available")
    
    def cleanup(self):
        """Cleanup resources"""
        # Stop evidence recording
        if hasattr(self, 'evidence_agent'):
            self.evidence_agent.force_stop_recording()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")
        
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print evidence agent status
        if hasattr(self, 'evidence_agent'):
            evidence_status = self.evidence_agent.get_status()
            print(f"\n=== EVIDENCE AGENT STATUS ===")
            print(f"Buffered Frames: {evidence_status['buffered_frames']}")
            print(f"Final Recording: {evidence_status['current_file'] or 'None'}")
            print(f"Total Recordings: Check evidence/videos/ folder")

def main():
    """Main entry point"""
    print("=" * 60)
    print("INTEGRATED GUN DETECTION SYSTEM")
    print("=" * 60)
    
    # Check if model exists
    import os
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} model not found!")
        print(f"Please ensure {model_path} is available")
        return
    
    try:
        # Initialize and run system
        system = IntegratedGunDetectionSystem(model_path=model_path)
        system.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
