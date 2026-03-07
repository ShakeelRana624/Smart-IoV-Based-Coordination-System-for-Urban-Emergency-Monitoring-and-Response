"""
Firebase Setup Guide for Intelligent Weapon Detection System

This guide explains how to set up Firebase for storing detection alerts.
"""

# FIREBASE SETUP INSTRUCTIONS

## 1. Create Firebase Project
1. Go to https://console.firebase.google.com/
2. Create a new project (e.g., "weapon-detection-system")
3. Enable Firestore Database
4. Enable Cloud Storage

## 2. Generate Service Account Key
1. Go to Project Settings > Service Accounts
2. Click "Generate new private key"
3. Download the JSON file
4. Rename it to `serviceAccountKey.json`
5. Place it in the project root directory

## 3. Update Firebase Configuration
In `utils/firebase_alert_storage.py`, update the storage bucket name:

```python
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-name.appspot.com'  # Update with your bucket name
})
```

## 4. Firestore Database Setup
Create the following collections in Firestore:
- `detection_alerts` - Stores individual alerts
- `alert_summaries` - Stores alert summaries
- `system_status` - Stores system status updates

## 5. Security Rules (Optional)
For Firestore:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if request.time < timestamp.date(2025, 1, 1);
    }
  }
}
```

For Storage:
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if request.time < timestamp.date(2025, 1, 1);
    }
  }
}
```

## 6. Test Firebase Connection
Run this test script:
```python
from utils.firebase_alert_storage import FirebaseAlertStorage

# Test Firebase connection
firebase_storage = FirebaseAlertStorage()
if firebase_storage.db:
    print("✅ Firebase connected successfully!")
else:
    print("❌ Firebase connection failed")
```

## 7. Firebase Data Structure

### Detection Alert Document
```json
{
  "alert_id": "uuid",
  "timestamp": "2026-03-07T11:57:02.367796",
  "camera_id": "CAM_001",
  "camera_location": "Main Security Camera",
  "detection_type": "WEAPON",
  "threat_level": "HIGH",
  "confidence": 0.63,
  "bbox": [186, 185, 120, 247],
  "description": "GUN detected with 0.63 confidence",
  "coordinates": {"x": 246, "y": 308, "width": 120, "height": 247},
  "emergency_state": "EMERGENCY",
  "additional_info": {"weapon_type": "Firearm", "frame_count": 35},
  "stored_at": "2026-03-07T11:57:02.369468",
  "processed": false
}
```

### Alert Summary Document
```json
{
  "summary_id": "uuid",
  "timestamp": "2026-03-07T11:57:02.369468",
  "camera_id": "CAM_001",
  "camera_location": "Main Security Camera",
  "total_alerts": 1,
  "threat_levels": {"HIGH": 1},
  "detection_types": {"WEAPON": 1},
  "emergency_state": "EMERGENCY",
  "alerts": [...],
  "stored_at": "2026-03-07T11:57:02.371000"
}
```

### System Status Document
```json
{
  "system_state": "EMERGENCY",
  "active_alerts": 1,
  "camera_status": "ACTIVE",
  "detection_count": 1,
  "frame_count": 35,
  "timestamp": "2026-03-07T11:57:02.369468",
  "updated_at": "2026-03-07T11:57:02.371000",
  "camera_id": "CAM_001"
}
```

## 8. Current System Status

### ✅ Working Features:
- **JSON Alert Generation**: Perfectly working
- **Alert Sound Effects**: BEEP BEEP BEEP sounds
- **Alert Summary**: Complete summary JSON
- **Firebase Integration Code**: Fully implemented
- **Firebase Dependencies**: Installed and ready

### ⚠️ Setup Required:
- **Firebase Project**: Create Firebase project
- **Service Account Key**: Generate and place serviceAccountKey.json
- **Storage Bucket**: Update bucket name in code
- **Database Rules**: Configure security rules

### 🎯 Integration Status:
- **Alert System**: 100% working
- **Firebase Code**: 100% integrated
- **JSON Format**: Perfect for Firebase
- **Real-time Storage**: Ready to store alerts

## 9. Alternative: Local JSON Storage

If Firebase setup is not available, the system can store alerts locally:

```python
import json
from datetime import datetime

# Local alert storage
def store_alert_locally(alert_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alerts/alert_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"✅ Alert stored locally: {filename}")
```

## 10. Production Deployment

For production deployment:
1. Set up Firebase project with proper security rules
2. Configure environment variables for sensitive data
3. Set up monitoring and logging
4. Configure backup and retention policies
5. Set up real-time alerts and notifications

---

**Note**: The current system is fully functional and ready for Firebase integration. 
JSON alerts are being generated perfectly - only Firebase setup is required to start storing them in the cloud.
