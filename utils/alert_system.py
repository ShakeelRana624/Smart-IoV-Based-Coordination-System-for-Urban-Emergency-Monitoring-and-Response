"""
Alert System for Intelligent Weapon Detection System

Generates JSON alerts for all detections including camera location, detection type, etc.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class DetectionAlert:
    """Detection alert data structure"""
    alert_id: str
    timestamp: str
    camera_id: str
    camera_location: str
    detection_type: str
    threat_level: str
    confidence: float
    bbox: List[int]
    description: str
    coordinates: Dict[str, float]
    emergency_state: str
    evidence_path: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class AlertSystem:
    """JSON Alert generation system for detections"""
    
    def __init__(self, camera_id: str = "CAM_001", camera_location: str = "Main Entrance"):
        self.camera_id = camera_id
        self.camera_location = camera_location
        self.alerts_history = []
        
    def create_weapon_alert(self, detection: Dict[str, Any], frame_count: int) -> DetectionAlert:
        """Create weapon detection alert"""
        class_name = detection.get("meta", {}).get("class_name", "Unknown")
        confidence = detection.get("meta", {}).get("raw_confidence", 0.0)
        bbox = detection.get("bbox", [0, 0, 0, 0])
        
        # Determine threat level
        if class_name in ["GUN", "KNIFE"]:
            threat_level = "HIGH"
        else:
            threat_level = "MEDIUM"
        
        return DetectionAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            detection_type="WEAPON",
            threat_level=threat_level,
            confidence=confidence,
            bbox=bbox,
            description=f"{class_name} detected with {confidence:.2f} confidence",
            coordinates={
                "x": bbox[0] + bbox[2]//2,
                "y": bbox[1] + bbox[3]//2,
                "width": bbox[2],
                "height": bbox[3]
            },
            emergency_state="EMERGENCY" if threat_level in ["HIGH", "CRITICAL"] else "THREAT_DETECTION",
            additional_info={
                "weapon_type": detection.get("meta", {}).get("weapon_type", "Unknown"),
                "frame_count": frame_count,
                "detection_id": detection.get("id", 0)
            }
        )
    
    def create_violence_alert(self, detection: Dict[str, Any], frame_count: int) -> DetectionAlert:
        """Create violence detection alert"""
        violence_confidence = detection.get("violence_confidence", 0.0)
        bbox = detection.get("bbox", [0, 0, 0, 0])
        
        return DetectionAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            detection_type="VIOLENCE",
            threat_level="HIGH",
            confidence=violence_confidence,
            bbox=bbox,
            description=f"Violence detected with {violence_confidence:.2f} confidence",
            coordinates={
                "x": bbox[0] + bbox[2]//2,
                "y": bbox[1] + bbox[3]//2,
                "width": bbox[2],
                "height": bbox[3]
            },
            emergency_state="EMERGENCY",
            additional_info={
                "person_id": detection.get("id", 0),
                "frame_count": frame_count,
                "violence_detected": detection.get("violence_detected", False)
            }
        )
    
    def create_fire_alert(self, fire_smoke_result: Dict[str, Any], frame_count: int) -> List[DetectionAlert]:
        """Create fire detection alerts"""
        alerts = []
        
        if fire_smoke_result.get("fire_detected", False):
            for fire_detection in fire_smoke_result.get("fire_detections", []):
                bbox = fire_detection.get("bbox", [0, 0, 0, 0])
                confidence = fire_detection.get("confidence", 0.0)
                
                alert = DetectionAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    camera_id=self.camera_id,
                    camera_location=self.camera_location,
                    detection_type="FIRE",
                    threat_level="CRITICAL",
                    confidence=confidence,
                    bbox=bbox,
                    description=f"Fire detected with {confidence:.2f} confidence",
                    coordinates={
                        "x": bbox[0] + bbox[2]//2,
                        "y": bbox[1] + bbox[3]//2,
                        "width": bbox[2],
                        "height": bbox[3]
                    },
                    emergency_state="EMERGENCY",
                    additional_info={
                        "fire_id": fire_detection.get("id", 0),
                        "frame_count": frame_count,
                        "total_fires": fire_smoke_result.get("fire_count", 0)
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def create_smoke_alert(self, fire_smoke_result: Dict[str, Any], frame_count: int) -> List[DetectionAlert]:
        """Create smoke detection alerts"""
        alerts = []
        
        if fire_smoke_result.get("smoke_detected", False):
            for smoke_detection in fire_smoke_result.get("smoke_detections", []):
                bbox = smoke_detection.get("bbox", [0, 0, 0, 0])
                confidence = smoke_detection.get("confidence", 0.0)
                
                alert = DetectionAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    camera_id=self.camera_id,
                    camera_location=self.camera_location,
                    detection_type="SMOKE",
                    threat_level="HIGH",
                    confidence=confidence,
                    bbox=bbox,
                    description=f"Smoke detected with {confidence:.2f} confidence",
                    coordinates={
                        "x": bbox[0] + bbox[2]//2,
                        "y": bbox[1] + bbox[3]//2,
                        "width": bbox[2],
                        "height": bbox[3]
                    },
                    emergency_state="EMERGENCY",
                    additional_info={
                        "smoke_id": smoke_detection.get("id", 0),
                        "frame_count": frame_count,
                        "total_smokes": fire_smoke_result.get("smoke_count", 0)
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def create_pose_alert(self, detection: Dict[str, Any], frame_count: int, pose_type: str) -> DetectionAlert:
        """Create suspicious pose detection alert"""
        bbox = detection.get("bbox", [0, 0, 0, 0])
        confidence = 0.8  # Default confidence for pose detection
        
        return DetectionAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            detection_type="SUSPICIOUS_POSE",
            threat_level="MEDIUM",
            confidence=confidence,
            bbox=bbox,
            description=f"Suspicious {pose_type} pose detected",
            coordinates={
                "x": bbox[0] + bbox[2]//2,
                "y": bbox[1] + bbox[3]//2,
                "width": bbox[2],
                "height": bbox[3]
            },
            emergency_state="SUSPICIOUS",
            additional_info={
                "pose_type": pose_type,
                "person_id": detection.get("id", 0),
                "frame_count": frame_count
            }
        )
    
    def create_system_alert(self, alert_type: str, message: str, threat_level: str = "LOW") -> DetectionAlert:
        """Create system-level alert"""
        return DetectionAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            detection_type="SYSTEM",
            threat_level=threat_level,
            confidence=1.0,
            bbox=[0, 0, 0, 0],
            description=message,
            coordinates={"x": 0, "y": 0, "width": 0, "height": 0},
            emergency_state="NORMAL",
            additional_info={
                "alert_type": alert_type,
                "system_message": True
            }
        )
    
    def alert_to_json(self, alert: DetectionAlert) -> str:
        """Convert alert to JSON string"""
        return json.dumps(asdict(alert), indent=2)
    
    def create_alert_summary(self, alerts: List[DetectionAlert]) -> Dict[str, Any]:
        """Create summary of all alerts"""
        summary = {
            "summary_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "camera_id": self.camera_id,
            "camera_location": self.camera_location,
            "total_alerts": len(alerts),
            "threat_levels": {},
            "detection_types": {},
            "emergency_state": "NORMAL",
            "alerts": []
        }
        
        # Count threat levels and detection types
        for alert in alerts:
            # Count threat levels
            threat_level = alert.threat_level
            summary["threat_levels"][threat_level] = summary["threat_levels"].get(threat_level, 0) + 1
            
            # Count detection types
            detection_type = alert.detection_type
            summary["detection_types"][detection_type] = summary["detection_types"].get(detection_type, 0) + 1
            
            # Determine highest emergency state
            if alert.emergency_state == "EMERGENCY":
                summary["emergency_state"] = "EMERGENCY"
            elif alert.emergency_state == "THREAT_DETECTION" and summary["emergency_state"] != "EMERGENCY":
                summary["emergency_state"] = "THREAT_DETECTION"
            elif alert.emergency_state == "SUSPICIOUS" and summary["emergency_state"] not in ["EMERGENCY", "THREAT_DETECTION"]:
                summary["emergency_state"] = "SUSPICIOUS"
            
            # Add alert to list
            summary["alerts"].append(asdict(alert))
        
        return summary
    
    def print_alert_json(self, alert: DetectionAlert, alert_sound: str = "🚨 BEEP BEEP BEEP 🚨"):
        """Print alert JSON with sound effect"""
        print("\n" + "="*80)
        print(alert_sound)
        print("="*80)
        print(self.alert_to_json(alert))
        print("="*80)
        print(alert_sound)
        print("="*80 + "\n")
        
        # Store in history
        self.alerts_history.append(alert)
    
    def print_summary_json(self, alerts: List[DetectionAlert]):
        """Print summary of all alerts as JSON"""
        if not alerts:
            return
            
        summary = self.create_alert_summary(alerts)
        
        print("\n" + "="*100)
        print("📊 ALERT SUMMARY JSON 📊")
        print("="*100)
        print(json.dumps(summary, indent=2))
        print("="*100 + "\n")
    
    def get_recent_alerts(self, count: int = 10) -> List[DetectionAlert]:
        """Get recent alerts"""
        return self.alerts_history[-count:] if self.alerts_history else []
    
    def clear_alert_history(self):
        """Clear alert history"""
        self.alerts_history.clear()

# Global alert system instance
alert_system = AlertSystem()

# Example usage
if __name__ == "__main__":
    # Example weapon detection alert
    weapon_detection = {
        "id": 1,
        "bbox": [100, 100, 50, 100],
        "meta": {
            "class_name": "GUN",
            "weapon_type": "Firearm",
            "raw_confidence": 0.85
        }
    }
    
    # Create and print weapon alert
    weapon_alert = alert_system.create_weapon_alert(weapon_detection, 100)
    alert_system.print_alert_json(weapon_alert)
    
    # Example fire detection alert
    fire_result = {
        "fire_detected": True,
        "fire_count": 1,
        "fire_detections": [
            {
                "id": 1,
                "bbox": [200, 200, 80, 60],
                "confidence": 0.92
            }
        ]
    }
    
    # Create and print fire alerts
    fire_alerts = alert_system.create_fire_alert(fire_result, 101)
    for alert in fire_alerts:
        alert_system.print_alert_json(alert)
    
    # Print summary
    all_alerts = [weapon_alert] + fire_alerts
    alert_system.print_summary_json(all_alerts)
