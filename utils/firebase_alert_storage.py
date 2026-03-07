"""
Firebase Alert Storage System for Intelligent Weapon Detection System

Stores JSON alerts in Firebase Firestore for real-time monitoring and analysis.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import uuid
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import os
from config.firebase_config import firebase_config_manager

class FirebaseAlertStorage:
    """Firebase storage system for detection alerts"""
    
    def __init__(self):
        """Initialize Firebase with configuration from config manager"""
        # Load configuration
        self.config = firebase_config_manager.load_config()
        
        # Check if service account key exists
        service_account_path = firebase_config_manager.get_service_account_path()
        
        try:
            if not os.path.exists(service_account_path):
                print(f"❌ Service account key not found: {service_account_path}")
                print("📁 Please download your service account key from Firebase Console")
                print("📁 Place it in the project root directory as 'serviceAccountKey.json'")
                print("📁 Or run 'python setup_firebase.py' to configure Firebase")
                raise FileNotFoundError("Service account key not found")
            
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': firebase_config_manager.get_storage_bucket()
            })
            
            # Initialize Firestore
            self.db = firestore.client()
            self.storage = storage.bucket()
            
            # Get collection names from config
            collections = firebase_config_manager.get_collections()
            self.alerts_collection = self.db.collection(collections['alerts'])
            self.alerts_summary_collection = self.db.collection(collections['summaries'])
            self.system_status_collection = self.db.collection(collections['status'])
            
            self.firebase_available = True
            print("✓ Firebase initialized successfully")
            print("✓ Firestore database connected")
            print("✓ Storage bucket connected")
            print(f"✓ Project: {firebase_config_manager.get_project_id()}")
            
        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            print("📁 Using local JSON storage instead")
            print("📁 Alerts and evidence will be stored locally")
            print("📁 To enable Firebase storage:")
            print("   1. Create Firebase project")
            print("   2. Download service account key")
            print("   3. Place serviceAccountKey.json in project root")
            print("   4. Run 'python setup_firebase.py'")
            
            self.db = None
            self.storage = None
            self.firebase_available = False
            
            # Create local storage directories from config
            local_config = firebase_config_manager.get_local_storage_config()
            os.makedirs(local_config['alerts_dir'], exist_ok=True)
            os.makedirs(local_config['summaries_dir'], exist_ok=True)
            os.makedirs(local_config['status_dir'], exist_ok=True)
            os.makedirs(local_config['evidence_dir'], exist_ok=True)
            
            print(f"✓ Local storage directories created")
    
    def store_alert(self, alert_data: Dict[str, Any]) -> Optional[str]:
        """Store single alert in Firebase or local storage"""
        if self.firebase_available and self.db:
            try:
                # Generate document ID
                alert_id = alert_data.get('alert_id', str(uuid.uuid4()))
                
                # Add metadata
                alert_data['stored_at'] = datetime.now().isoformat()
                alert_data['processed'] = False
                
                # Store in Firestore
                doc_ref = self.alerts_collection.document(alert_id)
                doc_ref.set(alert_data)
                
                print(f"✓ Alert stored in Firebase: {alert_id}")
                return alert_id
                
            except Exception as e:
                print(f"❌ Failed to store alert in Firebase: {e}")
                print("📁 Storing locally instead")
        
        # Local storage fallback
        try:
            alert_id = alert_data.get('alert_id', str(uuid.uuid4()))
            alert_data['stored_at'] = datetime.now().isoformat()
            alert_data['processed'] = False
            alert_data['storage_type'] = 'local'
            
            # Get local storage config
            local_config = firebase_config_manager.get_local_storage_config()
            
            # Store locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{local_config['alerts_dir']}/alert_{timestamp}_{alert_id[:8]}.json"
            
            with open(filename, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            print(f"✓ Alert stored locally: {filename}")
            return alert_id
            
        except Exception as e:
            print(f"❌ Failed to store alert locally: {e}")
            return None
    
    def store_alert_summary(self, summary_data: Dict[str, Any]) -> Optional[str]:
        """Store alert summary in Firebase or local storage"""
        if self.firebase_available and self.db:
            try:
                # Generate document ID
                summary_id = summary_data.get('summary_id', str(uuid.uuid4()))
                
                # Add metadata
                summary_data['stored_at'] = datetime.now().isoformat()
                summary_data['processed'] = False
                
                # Store in Firestore
                doc_ref = self.alerts_summary_collection.document(summary_id)
                doc_ref.set(summary_data)
                
                print(f"✓ Alert summary stored in Firebase: {summary_id}")
                return summary_id
                
            except Exception as e:
                print(f"❌ Failed to store alert summary in Firebase: {e}")
                print("📁 Storing locally instead")
        
        # Local storage fallback
        try:
            summary_id = summary_data.get('summary_id', str(uuid.uuid4()))
            summary_data['stored_at'] = datetime.now().isoformat()
            summary_data['processed'] = False
            summary_data['storage_type'] = 'local'
            
            # Get local storage config
            local_config = firebase_config_manager.get_local_storage_config()
            
            # Store locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{local_config['summaries_dir']}/summary_{timestamp}_{summary_id[:8]}.json"
            
            with open(filename, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"✓ Alert summary stored locally: {filename}")
            return summary_id
            
        except Exception as e:
            print(f"❌ Failed to store alert summary locally: {e}")
            return None
    
    def store_multiple_alerts(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Store multiple alerts in Firebase using batch"""
        if not self.db:
            print("❌ Firebase not initialized")
            return []
        
        try:
            # Create batch operation
            batch = self.db.batch()
            stored_ids = []
            
            for alert_data in alerts:
                # Generate document ID
                alert_id = alert_data.get('alert_id', str(uuid.uuid4()))
                
                # Add metadata
                alert_data['stored_at'] = datetime.now().isoformat()
                alert_data['processed'] = False
                
                # Add to batch
                doc_ref = self.alerts_collection.document(alert_id)
                batch.set(doc_ref, alert_data)
                stored_ids.append(alert_id)
            
            # Commit batch
            batch.commit()
            
            print(f"✓ {len(alerts)} alerts stored in Firebase")
            return stored_ids
            
        except Exception as e:
            print(f"❌ Failed to store multiple alerts: {e}")
            return []
    
    def store_evidence_file(self, file_path: str, alert_id: str, detection_type: str = "WEAPON") -> Optional[str]:
        """Store evidence file in Firebase Storage or local storage"""
        if self.firebase_available and self.storage:
            try:
                # Generate blob name with detection type
                file_name = os.path.basename(file_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                blob_name = f"evidence/{detection_type.lower()}/{timestamp}_{alert_id}_{file_name}"
                
                # Upload file
                blob = self.storage.blob(blob_name)
                blob.upload_from_filename(file_path)
                
                # Make file public
                blob.make_public()
                
                # Get public URL
                file_url = blob.public_url
                
                # Store evidence metadata in Firestore
                evidence_metadata = {
                    "evidence_id": str(uuid.uuid4()),
                    "alert_id": alert_id,
                    "detection_type": detection_type,
                    "file_name": file_name,
                    "blob_name": blob_name,
                    "public_url": file_url,
                    "file_size": os.path.getsize(file_path),
                    "timestamp": datetime.now().isoformat(),
                    "stored_at": datetime.now().isoformat(),
                    "storage_type": "firebase"
                }
                
                # Store in evidence collection
                collections = firebase_config_manager.get_collections()
                evidence_collection = self.db.collection(collections['evidence'])
                evidence_collection.add(evidence_metadata)
                
                print(f"✓ Evidence file stored in Firebase: {file_url}")
                return file_url
                
            except Exception as e:
                print(f"❌ Failed to store evidence file in Firebase: {e}")
                print("📁 Storing evidence locally instead")
        
        # Local storage fallback
        try:
            # Get local storage config
            local_config = firebase_config_manager.get_local_storage_config()
            
            # Generate local filename
            file_name = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_filename = f"{timestamp}_{alert_id}_{file_name}"
            local_path = os.path.join(local_config['evidence_dir'], local_filename)
            
            # Copy file to local evidence directory
            import shutil
            shutil.copy2(file_path, local_path)
            
            # Store evidence metadata locally
            evidence_metadata = {
                "evidence_id": str(uuid.uuid4()),
                "alert_id": alert_id,
                "detection_type": detection_type,
                "file_name": file_name,
                "local_path": local_path,
                "file_size": os.path.getsize(file_path),
                "timestamp": datetime.now().isoformat(),
                "stored_at": datetime.now().isoformat(),
                "storage_type": "local"
            }
            
            # Store metadata in local JSON file
            metadata_filename = f"{local_config['evidence_dir']}/metadata_{timestamp}_{alert_id[:8]}.json"
            with open(metadata_filename, 'w') as f:
                json.dump(evidence_metadata, f, indent=2)
            
            print(f"✓ Evidence file stored locally: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Failed to store evidence file locally: {e}")
            return None
    
    def update_system_status(self, status_data: Dict[str, Any]) -> Optional[str]:
        """Update system status in Firebase or local storage"""
        if self.firebase_available and self.db:
            try:
                # Generate document ID with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                status_id = f"status_{timestamp}"
                
                # Add metadata
                status_data['updated_at'] = datetime.now().isoformat()
                status_data['camera_id'] = status_data.get('camera_id', 'CAM_001')
                
                # Store in Firestore
                doc_ref = self.system_status_collection.document(status_id)
                doc_ref.set(status_data)
                
                print(f"✓ System status updated in Firebase: {status_id}")
                return status_id
                
            except Exception as e:
                print(f"❌ Failed to update system status in Firebase: {e}")
                print("📁 Storing locally instead")
        
        # Local storage fallback
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_id = f"status_{timestamp}"
            
            status_data['updated_at'] = datetime.now().isoformat()
            status_data['camera_id'] = status_data.get('camera_id', 'CAM_001')
            status_data['storage_type'] = 'local'
            
            # Get local storage config
            local_config = firebase_config_manager.get_local_storage_config()
            
            # Store locally
            filename = f"{local_config['status_dir']}/status_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            print(f"✓ System status stored locally: {filename}")
            return status_id
            
        except Exception as e:
            print(f"❌ Failed to store system status locally: {e}")
            return None
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from Firebase"""
        if not self.db:
            print("❌ Firebase not initialized")
            return []
        
        try:
            # Query recent alerts
            alerts = self.alerts_collection.order_by(
                'timestamp', direction=firestore.DESCENDING
            ).limit(limit).get()
            
            alert_list = []
            for doc in alerts:
                alert_data = doc.to_dict()
                alert_data['id'] = doc.id
                alert_list.append(alert_data)
            
            print(f"✓ Retrieved {len(alert_list)} recent alerts")
            return alert_list
            
        except Exception as e:
            print(f"❌ Failed to get recent alerts: {e}")
            return []
    
    def get_alerts_by_type(self, detection_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get alerts by detection type"""
        if not self.db:
            print("❌ Firebase not initialized")
            return []
        
        try:
            # Query alerts by type
            alerts = self.alerts_collection.where(
                'detection_type', '==', detection_type
            ).order_by('timestamp', direction=firestore.DESCENDING).limit(limit).get()
            
            alert_list = []
            for doc in alerts:
                alert_data = doc.to_dict()
                alert_data['id'] = doc.id
                alert_list.append(alert_data)
            
            print(f"✓ Retrieved {len(alert_list)} {detection_type} alerts")
            return alert_list
            
        except Exception as e:
            print(f"❌ Failed to get alerts by type: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics from Firebase"""
        if not self.db:
            print("❌ Firebase not initialized")
            return {}
        
        try:
            # Get all alerts for statistics
            all_alerts = self.alerts_collection.get()
            
            stats = {
                'total_alerts': 0,
                'detection_types': {},
                'threat_levels': {},
                'emergency_states': {},
                'recent_activity': []
            }
            
            for doc in all_alerts:
                alert_data = doc.to_dict()
                stats['total_alerts'] += 1
                
                # Count detection types
                detection_type = alert_data.get('detection_type', 'Unknown')
                stats['detection_types'][detection_type] = stats['detection_types'].get(detection_type, 0) + 1
                
                # Count threat levels
                threat_level = alert_data.get('threat_level', 'Unknown')
                stats['threat_levels'][threat_level] = stats['threat_levels'].get(threat_level, 0) + 1
                
                # Count emergency states
                emergency_state = alert_data.get('emergency_state', 'Unknown')
                stats['emergency_states'][emergency_state] = stats['emergency_states'].get(emergency_state, 0) + 1
            
            # Get recent activity (last 5)
            recent_alerts = self.alerts_collection.order_by(
                'timestamp', direction=firestore.DESCENDING
            ).limit(5).get()
            
            for doc in recent_alerts:
                alert_data = doc.to_dict()
                stats['recent_activity'].append({
                    'id': doc.id,
                    'type': alert_data.get('detection_type', 'Unknown'),
                    'timestamp': alert_data.get('timestamp', ''),
                    'description': alert_data.get('description', '')
                })
            
            print(f"✓ Generated alert statistics: {stats['total_alerts']} total alerts")
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get alert statistics: {e}")
            return {}
    
    def mark_alert_processed(self, alert_id: str) -> bool:
        """Mark alert as processed"""
        if not self.db:
            print("❌ Firebase not initialized")
            return False
        
        try:
            # Update alert
            self.alerts_collection.document(alert_id).update({
                'processed': True,
                'processed_at': datetime.now().isoformat()
            })
            
            print(f"✓ Alert marked as processed: {alert_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to mark alert as processed: {e}")
            return False
    
    def delete_old_alerts(self, days_old: int = 30) -> int:
        """Delete alerts older than specified days"""
        if not self.db:
            print("❌ Firebase not initialized")
            return 0
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            # Get old alerts
            old_alerts = self.alerts_collection.where(
                'timestamp', '<', cutoff_iso
            ).get()
            
            # Delete old alerts
            deleted_count = 0
            for doc in old_alerts:
                doc.reference.delete()
                deleted_count += 1
            
            print(f"✓ Deleted {deleted_count} alerts older than {days_old} days")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Failed to delete old alerts: {e}")
            return 0

# Global Firebase storage instance
firebase_storage = FirebaseAlertStorage()

# Example usage
if __name__ == "__main__":
    # Example alert data
    example_alert = {
        "alert_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "camera_id": "CAM_001",
        "camera_location": "Main Security Camera",
        "detection_type": "WEAPON",
        "threat_level": "HIGH",
        "confidence": 0.85,
        "bbox": [100, 100, 50, 100],
        "description": "GUN detected with 0.85 confidence",
        "coordinates": {"x": 125, "y": 150, "width": 50, "height": 100},
        "emergency_state": "EMERGENCY",
        "additional_info": {"weapon_type": "Firearm", "frame_count": 100}
    }
    
    # Store alert
    alert_id = firebase_storage.store_alert(example_alert)
    
    # Get recent alerts
    recent_alerts = firebase_storage.get_recent_alerts(limit=5)
    
    # Get statistics
    stats = firebase_storage.get_alert_statistics()
    
    print(f"Firebase Alert Storage System Test Complete")
    print(f"Stored alert: {alert_id}")
    print(f"Recent alerts: {len(recent_alerts)}")
    print(f"Statistics: {stats}")
