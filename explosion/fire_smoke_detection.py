"""
Fire and Smoke Detection Module for Intelligent Weapon Detection System

Detects fire and smoke using fire-smoke.pt model and integrates with main system.
Shows fire/smoke detection with red bounding boxes and triggers emergency responses.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Any, Tuple, Optional
import time
import os

class FireSmokeDetector:
    """Fire and smoke detection using fire-smoke.pt model"""
    
    def __init__(self, model_path: str = "models/fire-smoke.pt"):
        """
        Initialize fire and smoke detector
        
        Args:
            model_path: Path to fire-smoke detection model
        """
        try:
            # Load the fire-smoke detection model using YOLOv8
            self.model = YOLO(model_path)
            self.model.conf = 0.5  # Set confidence threshold
            print(f"✓ Fire-Smoke model loaded: {model_path}")
            
            # Get model classes
            self.classes = self.model.names
            print(f"🔥 Fire-Smoke Classes: {self.classes}")
            
        except Exception as e:
            print(f"❌ Failed to load fire-smoke model: {e}")
            print("📁 Please ensure 'fire-smoke.pt' is in the models folder")
            print("📁 Fire-smoke detection will be disabled")
            self.model = None
            self.classes = {}
        
        # Fire and smoke detection parameters
        self.confidence_threshold = 0.5
        self.fire_detected = False
        self.smoke_detected = False
        self.last_detection_time = 0
        self.detection_history = []
        
        # Storage for fire/smoke data
        self.detected_fires = {}  # {detection_id: fire_info}
        self.detected_smokes = {}  # {detection_id: smoke_info}
        self.detection_id_counter = 1
        
    def detect_fire_smoke_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect fire and smoke in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary: {detection_results}
        """
        if self.model is None:
            return {
                "fire_detected": False,
                "smoke_detected": False,
                "fire_count": 0,
                "smoke_count": 0,
                "detections": [],
                "confidence": 0.0
            }
        
        try:
            # Run YOLOv8 inference on the frame
            results = self.model(frame, conf=self.confidence_threshold)
            
            # Process detections
            fire_detections = []
            smoke_detections = []
            fire_count = 0
            smoke_count = 0
            max_confidence = 0.0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = float(box.conf[0])  # Confidence score
                    class_id = int(box.cls[0])     # Class ID
                    class_name = self.classes.get(class_id, "Unknown")
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                    
                    max_confidence = max(max_confidence, confidence)
                    
                    detection_info = {
                        "id": self.detection_id_counter,
                        "bbox": bbox,
                        "confidence": confidence,
                        "class_name": class_name,
                        "class_id": class_id,
                        "timestamp": time.time(),
                        "center": (int(x1 + (x2-x1)//2), int(y1 + (y2-y1)//2))
                    }
                    
                    # Check if it's fire or smoke
                    if "fire" in class_name.lower():
                        fire_count += 1
                        fire_detections.append(detection_info)
                        self.detected_fires[self.detection_id_counter] = detection_info
                        print(f"🔥 FIRE DETECTED: {class_name} with confidence {confidence:.2f}")
                        
                    elif "smoke" in class_name.lower():
                        smoke_count += 1
                        smoke_detections.append(detection_info)
                        self.detected_smokes[self.detection_id_counter] = detection_info
                        print(f"💨 SMOKE DETECTED: {class_name} with confidence {confidence:.2f}")
                    
                    self.detection_id_counter += 1
            
            # Update detection status
            self.fire_detected = fire_count > 0
            self.smoke_detected = smoke_count > 0
            self.last_detection_time = time.time()
            
            # Store detection history
            detection_result = {
                "fire_detected": self.fire_detected,
                "smoke_detected": self.smoke_detected,
                "fire_count": fire_count,
                "smoke_count": smoke_count,
                "fire_detections": fire_detections,
                "smoke_detections": smoke_detections,
                "all_detections": fire_detections + smoke_detections,
                "confidence": max_confidence,
                "timestamp": self.last_detection_time
            }
            
            self.detection_history.append(detection_result)
            self.detection_history = self.detection_history[-100:]  # Keep last 100 frames
            
            return detection_result
            
        except Exception as e:
            print(f"Error in fire-smoke detection: {e}")
            return {
                "fire_detected": False,
                "smoke_detected": False,
                "fire_count": 0,
                "smoke_count": 0,
                "detections": [],
                "confidence": 0.0
            }
    
    def draw_fire_smoke_on_frame(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw fire and smoke detection on frame with red bounding boxes
        
        Args:
            frame: Input frame
            detection_result: Fire and smoke detection results
            
        Returns:
            Frame with fire and smoke detection drawn
        """
        try:
            annotated = frame.copy()
            
            # Draw all detections
            for detection in detection_result.get("all_detections", []):
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                class_name = detection["class_name"]
                detection_id = detection["id"]
                
                x, y, w, h = bbox
                
                # Determine color based on class
                if "fire" in class_name.lower():
                    color = (0, 0, 255)  # Red for fire
                    symbol = "🔥"
                    label_prefix = "FIRE"
                elif "smoke" in class_name.lower():
                    color = (128, 128, 128)  # Gray for smoke
                    symbol = "💨"
                    label_prefix = "SMOKE"
                else:
                    color = (255, 255, 0)  # Yellow for unknown
                    symbol = "⚠️"
                    label_prefix = "DETECTION"
                
                # Draw thick bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
                
                # Draw detection label
                label = f"{symbol} {label_prefix} ID:{detection_id}"
                conf_text = f"Conf:{confidence:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated, (x, y - 40), (x + label_size[0] + 10, y - 10), color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, conf_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw summary information
            fire_count = detection_result.get("fire_count", 0)
            smoke_count = detection_result.get("smoke_count", 0)
            
            if fire_count > 0 or smoke_count > 0:
                # Draw summary box
                summary_text = f"🔥 Fire: {fire_count} | 💨 Smoke: {smoke_count}"
                summary_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Draw summary background
                cv2.rectangle(annotated, (10, 10), (summary_size[0] + 20, 50), (0, 0, 0), -1)
                cv2.rectangle(annotated, (10, 10), (summary_size[0] + 20, 50), (0, 0, 255), 2)
                
                # Draw summary text
                cv2.putText(annotated, summary_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add warning
                if fire_count > 0:
                    cv2.putText(annotated, "⚠️ FIRE EMERGENCY", (annotated.shape[1] // 2 - 100, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            return annotated
            
        except Exception as e:
            print(f"Error drawing fire-smoke detection: {e}")
            return frame
    
    def get_fire_detections(self) -> Dict[int, Dict[str, Any]]:
        """Get all detected fires"""
        return self.detected_fires.copy()
    
    def get_smoke_detections(self) -> Dict[int, Dict[str, Any]]:
        """Get all detected smokes"""
        return self.detected_smokes.copy()
    
    def get_fire_count(self) -> int:
        """Get count of currently detected fires"""
        return len(self.detected_fires)
    
    def get_smoke_count(self) -> int:
        """Get count of currently detected smokes"""
        return len(self.detected_smokes)
    
    def is_fire_detected(self) -> bool:
        """Check if any fire is currently detected"""
        return self.fire_detected
    
    def is_smoke_detected(self) -> bool:
        """Check if any smoke is currently detected"""
        return self.smoke_detected
    
    def is_emergency_detected(self) -> bool:
        """Check if fire or smoke emergency is detected"""
        return self.fire_detected or self.smoke_detected
    
    def clear_detections(self):
        """Clear all stored detections"""
        self.detected_fires.clear()
        self.detected_smokes.clear()
        self.fire_detected = False
        self.smoke_detected = False
        self.detection_history.clear()
        self.detection_id_counter = 1
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            "total_fires": len(self.detected_fires),
            "total_smokes": len(self.detected_smokes),
            "fire_detected": self.fire_detected,
            "smoke_detected": self.smoke_detected,
            "emergency_detected": self.is_emergency_detected(),
            "last_detection_time": self.last_detection_time,
            "detection_history_length": len(self.detection_history)
        }


# Real-time testing function
def test_fire_smoke_detection():
    """Test fire and smoke detection in real-time"""
    print("🔥 Starting Fire-Smoke Detection Test...")
    
    # Initialize detector
    detector = FireSmokeDetector()
    
    if detector.model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera. Exiting...")
        return
    
    print("📹 Camera opened successfully!")
    print("🔥 Fire-Smoke Detection Started...")
    print("Press 'q' to quit, 's' to save frame, 'c' to clear detections")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            break
        
        # Detect fire and smoke
        detection_result = detector.detect_fire_smoke_in_frame(frame)
        
        # Draw detections on frame
        annotated_frame = detector.draw_fire_smoke_on_frame(frame, detection_result)
        
        # Add system info
        info_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))} | Fire: {detection_result['fire_count']} | Smoke: {detection_result['smoke_count']}"
        cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("🔥 Fire-Smoke Detection", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fire_smoke_detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Frame saved: {filename}")
        elif key == ord('c'):
            # Clear detections
            detector.clear_detections()
            print("🧹 Detections cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fire-Smoke Detection Test Completed")


if __name__ == "__main__":
    # Run real-time test
    test_fire_smoke_detection()
