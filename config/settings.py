"""
Configuration Settings for Integrated Gun Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Model Configuration
MODEL_CONFIG = {
    "model_path": str(MODEL_DIR / "best.pt"),
    "confidence_threshold": 0.5,
    "input_resolution": (640, 480),
    "target_fps": 30
}

# Camera Configuration
CAMERA_CONFIG = {
    "camera_index": 0,
    "width": 640,
    "height": 480,
    "fps": 30
}

# Agent Configuration
AGENT_CONFIG = {
    "gun_threshold": 0.45,
    "knife_threshold": 0.4,
    "fight_threshold": 0.5,
    "ema_alpha": 0.4,
    "score_weights": {
        "severity": 0.6,
        "confidence": 0.25,
        "duration": 0.15
    },
    "critical_score": 3.2,
    "high_score": 2.4,
    "medium_score": 1.6
}

# Evidence Configuration
EVIDENCE_CONFIG = {
    "storage_path": str(DATA_DIR / "evidence"),
    "database_path": str(DATA_DIR / "evidence" / "detections.db"),
    "auto_save_threshold": 2.0,
    "max_clips_per_event": 3,
    "clip_duration": 10.0
}

# Alert Configuration
ALERT_CONFIG = {
    "audio_enabled": True,
    "min_alert_interval": 2.0,  # seconds
    "frequencies": {
        "low": 800,      # Hz
        "medium": 1000,   # Hz
        "high": 1500      # Hz
    },
    "durations": {
        "low": 200,      # ms
        "medium": 300,    # ms
        "high": 500       # ms
    }
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    "webhook_url": os.getenv("WEBHOOK_URL", ""),
    "uav_endpoint": os.getenv("UAV_ENDPOINT", ""),
    "email_enabled": False,
    "sms_enabled": False
}

# Firebase Configuration
FIREBASE_CONFIG = {
    "service_account_key": "serviceAccountKey.json",
    "storage_bucket": "weapon-detection-system.appspot.com",
    "project_id": "weapon-detection-system",
    "database_url": "https://weapon-detection-system-default-rtdb.firebaseio.com/",
    "collections": {
        "alerts": "detection_alerts",
        "summaries": "alert_summaries",
        "status": "system_status",
        "evidence": "evidence_files"
    },
    "local_storage": {
        "enabled": True,
        "alerts_dir": "firebase_alerts",
        "summaries_dir": "firebase_summaries",
        "status_dir": "firebase_status",
        "evidence_dir": "firebase_evidence"
    },
    "auto_cleanup": {
        "enabled": True,
        "days_to_keep": 30,
        "max_files_per_dir": 1000
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": str(DATA_DIR / "logs" / "system.log")
}

# Threat Levels
THREAT_LEVELS = {
    "NONE": 0.0,
    "LOW": 1.0,
    "MEDIUM": 2.0,
    "HIGH": 3.0,
    "CRITICAL": 4.0
}

THREAT_STATES = {
    "NORMAL": "NORMAL",
    "SUSPICIOUS": "SUSPICIOUS", 
    "ARMED": "ARMED",
    "VIOLENT": "VIOLENT",
    "CRITICAL": "CRITICAL"
}
