"""
Violence Detection Module for Intelligent Weapon Detection System

Detects violent behavior using violence.pt model and integrates with person tracking.
Shows violence detection with red bounding boxes and person IDs.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import time
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_system import get_logger
from utils.error_handling import handle_errors, DetectionError, ModelError

class ViolenceDetector:
    """Violence detection using violence.pt model"""
    
    def __init__(self, model_path: str = "models/violence.pth"):
        """
        Initialize violence detector
        
        Args:
            model_path: Path to violence detection model
        """
        self.logger = get_logger()
        self.model_path = model_path
        
        # Violence detection parameters
        self.confidence_threshold = 0.7
        self.violence_sequence_length = 30  # Number of frames to analyze
        self.violence_detection_frames = []  # Buffer for violence detection
        
        # Storage for violence data
        self.detected_violence = {}  # {person_id: violence_info}
        self.violence_active = False
        self.last_violence_time = 0
        
        # Model input size (adjust based on your model requirements)
        self.input_size = (224, 224)  # Common size for violence detection models
        
        # Initialize model
        self.model = None
        self._load_model()
    
    @handle_errors(default_return=False, log_errors=True)
    def _load_model(self) -> bool:
        """Load violence detection model with proper error handling"""
        try:
            # Load the violence detection model using YOLOv8 instead of YOLOv5
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.conf = self.confidence_threshold  # Set confidence threshold
            
            self.logger.info(f"Violence model loaded successfully", extra={
                "component": "ViolenceDetector",
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold
            })
            return True
            
        except FileNotFoundError:
            raise ModelError(f"Violence model not found: {self.model_path}", 
                            model_name=self.model_path)
        except Exception as e:
            raise ModelError(f"Failed to load violence model: {str(e)}", 
                            model_name=self.model_path)
        
    def preprocess_frame_for_violence_detection(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Preprocess frame region for violence detection model
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Preprocessed frame ready for model input
        """
        try:
            x, y, w, h = bbox
            
            # Extract person region
            person_region = frame[y:y+h, x:x+w]
            
            if person_region.size == 0:
                return None
            
            # Resize to model input size (YOLOv5 handles preprocessing internally)
            person_region = cv2.resize(person_region, (640, 640))
            
            return person_region
            
        except Exception as e:
            print(f"Error preprocessing frame for violence detection: {e}")
            return None
    
    def detect_violence_in_region(self, frame: np.ndarray, bbox: List[int]) -> Tuple[bool, float]:
        """
        Detect violence in a specific region
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Tuple: (is_violence, confidence)
        """
        if self.model is None:
            return False, 0.0
        
        try:
            # Extract person region
            x, y, w, h = bbox
            person_region = frame[y:y+h, x:x+w]
            
            if person_region.size == 0:
                return False, 0.0
            
            # Run YOLOv8 inference on the person region
            results = self.model(person_region, conf=self.confidence_threshold)
            
            # Check if violence is detected
            violence_detected = False
            max_confidence = 0.0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = float(box.conf[0])  # Confidence score
                    class_id = int(box.cls[0])     # Class ID
                    
                    # Class 1 is 'Violence' in your model
                    if class_id == 1 and confidence > self.confidence_threshold:
                        violence_detected = True
                        max_confidence = max(max_confidence, confidence)
            
            return violence_detected, max_confidence
            
        except Exception as e:
            print(f"Error in violence detection: {e}")
            return False, 0.0
    
    def detect_violence_in_frame(self, frame: np.ndarray, person_detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Detect violence for all persons in frame
        
        Args:
            frame: Input frame
            person_detections: List of person detections with bounding boxes and IDs
            
        Returns:
            Dictionary: {person_id: violence_info}
        """
        if self.model is None:
            return {}
        
        violence_results = {}
        current_time = time.time()
        
        try:
            # Detect violence for each person
            for person_detection in person_detections:
                person_id = person_detection.get("id")
                person_bbox = person_detection.get("bbox", [])
                
                if not person_id or len(person_bbox) < 4:
                    continue
                
                # Detect violence in person's region
                is_violence, confidence = self.detect_violence_in_region(frame, person_bbox)
                
                # Store violence detection
                violence_info = {
                    "person_id": person_id,
                    "violence_detected": is_violence,
                    "confidence": confidence,
                    "bbox": person_bbox,
                    "timestamp": current_time
                }
                
                violence_results[person_id] = violence_info
                
                # Update storage
                self.detected_violence[person_id] = violence_info
                
                # Log violence detection info
                if is_violence:
                    self.logger.detection_event("violence", confidence, "system", person_id=person_id)
                    
                    # Update global violence status
                    self.violence_active = True
                    self.last_violence_time = current_time
                
                elif person_id in self.detected_violence and self.detected_violence[person_id].get("violence_detected", False):
                    # Violence was detected before but not now
                    self.logger.info(f"Violence ended for person {person_id}", extra={
                        "component": "ViolenceDetector",
                        "person_id": person_id,
                        "event_type": "violence_ended"
                    })
        
        except Exception as e:
            self.logger.error("Error in violence detection", exception=e, extra={
                "component": "ViolenceDetector"
            })
        
        # Check if any violence is still active
        if not any(info.get("violence_detected", False) for info in violence_results.values()):
            if self.violence_active:
                self.logger.info("All violence ended", extra={
                    "component": "ViolenceDetector",
                    "event_type": "all_violence_ended"
                })
                self.violence_active = False
        
        return violence_results
    
    def get_violence_info(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get violence information for a specific person
        
        Args:
            person_id: Person ID
            
        Returns:
            Violence information or None
        """
        return self.detected_violence.get(person_id)
    
    def get_all_violence(self) -> Dict[int, Dict[str, Any]]:
        """Get all detected violence"""
        return self.detected_violence.copy()
    
    def get_violence_count(self) -> int:
        """Get count of persons currently violent"""
        count = 0
        for violence_info in self.detected_violence.values():
            if violence_info.get("violence_detected", False):
                count += 1
        return count
    
    def get_violent_person_ids(self) -> List[int]:
        """Get list of person IDs currently violent"""
        ids = []
        for person_id, violence_info in self.detected_violence.items():
            if violence_info.get("violence_detected", False):
                ids.append(person_id)
        return ids
    
    def is_violence_active(self) -> bool:
        """Check if any violence is currently active"""
        return self.violence_active
    
    def clear_violence(self):
        """Clear all stored violence"""
        self.detected_violence.clear()
        self.violence_active = False
        self.violence_detection_frames.clear()
    
    def draw_violence_on_frame(self, frame: np.ndarray, violence_info: Dict[str, Any]) -> np.ndarray:
        """
        Draw violence detection on frame with red bounding box
        
        Args:
            frame: Input frame
            violence_info: Violence information dictionary
            
        Returns:
            Frame with violence detection drawn
        """
        try:
            person_id = violence_info.get("person_id")
            violence_detected = violence_info.get("violence_detected", False)
            confidence = violence_info.get("confidence", 0.0)
            bbox = violence_info.get("bbox", [])
            
            if not bbox or len(bbox) < 4:
                return frame
            
            x, y, w, h = bbox[:4]
            
            if violence_detected:
                # Draw red bounding box for violence
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Draw violence label
                violence_label = f"🥊 VIOLENCE ID:{person_id}"
                conf_text = f"Conf:{confidence:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(violence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y - 10), (0, 0, 255), -1)
                
                # Draw label text
                cv2.putText(frame, violence_label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, conf_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Add warning indicators
                cv2.putText(frame, "⚠️ VIOLENCE DETECTED", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing violence detection: {e}")
            return frame
    
    def update_violence_statistics(self, stats: Dict[str, Any]):
        """Update statistics with violence detection information"""
        if "violence_detections" not in stats:
            stats["violence_detections"] = 0
        
        current_violence_count = self.get_violence_count()
        if current_violence_count > stats.get("violence_detections", 0):
            stats["violence_detections"] = current_violence_count
