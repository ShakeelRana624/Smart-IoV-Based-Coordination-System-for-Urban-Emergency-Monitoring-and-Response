"""
Firebase Configuration Manager for Intelligent Weapon Detection System

Centralized Firebase configuration management with easy setup and switching.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import FIREBASE_CONFIG, BASE_DIR

class FirebaseConfigManager:
    """Manages Firebase configuration and setup"""
    
    def __init__(self):
        self.config = FIREBASE_CONFIG
        self.base_dir = BASE_DIR
        self.config_file = self.base_dir / "config" / "firebase_config.json"
        
    def load_config(self) -> Dict[str, Any]:
        """Load Firebase configuration from file or use defaults"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    config = {**self.config, **file_config}
                    print(f"✓ Firebase config loaded from: {self.config_file}")
                    return config
            else:
                print("📝 Using default Firebase config")
                return self.config.copy()
        except Exception as e:
            print(f"❌ Error loading Firebase config: {e}")
            return self.config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save Firebase configuration to file"""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ Firebase config saved to: {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving Firebase config: {e}")
            return False
    
    def get_service_account_path(self) -> str:
        """Get full path to service account key"""
        key_path = self.config.get("service_account_key", "serviceAccountKey.json")
        # If relative path, make it absolute from project root
        if not os.path.isabs(key_path):
            key_path = str(self.base_dir / key_path)
        return key_path
    
    def is_service_account_available(self) -> bool:
        """Check if service account key file exists"""
        return os.path.exists(self.get_service_account_path())
    
    def get_storage_bucket(self) -> str:
        """Get Firebase storage bucket name"""
        return self.config.get("storage_bucket", "weapon-detection-system.appspot.com")
    
    def get_project_id(self) -> str:
        """Get Firebase project ID"""
        return self.config.get("project_id", "weapon-detection-system")
    
    def get_database_url(self) -> str:
        """Get Firebase database URL"""
        return self.config.get("database_url", "")
    
    def get_collections(self) -> Dict[str, str]:
        """Get Firebase collection names"""
        return self.config.get("collections", {
            "alerts": "detection_alerts",
            "summaries": "alert_summaries", 
            "status": "system_status",
            "evidence": "evidence_files"
        })
    
    def get_local_storage_config(self) -> Dict[str, Any]:
        """Get local storage configuration"""
        return self.config.get("local_storage", {
            "enabled": True,
            "alerts_dir": "firebase_alerts",
            "summaries_dir": "firebase_summaries",
            "status_dir": "firebase_status",
            "evidence_dir": "firebase_evidence"
        })
    
    def is_local_storage_enabled(self) -> bool:
        """Check if local storage is enabled"""
        return self.config.get("local_storage", {}).get("enabled", True)
    
    def get_auto_cleanup_config(self) -> Dict[str, Any]:
        """Get auto cleanup configuration"""
        return self.config.get("auto_cleanup", {
            "enabled": True,
            "days_to_keep": 30,
            "max_files_per_dir": 1000
        })
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update specific configuration values"""
        try:
            current_config = self.load_config()
            
            # Deep update for nested dictionaries
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(current_config, updates)
            return self.save_config(current_config)
        except Exception as e:
            print(f"❌ Error updating Firebase config: {e}")
            return False
    
    def setup_firebase_project(self, project_id: str, storage_bucket: str = None, database_url: str = None) -> bool:
        """Setup Firebase project configuration"""
        updates = {
            "project_id": project_id
        }
        
        if storage_bucket:
            updates["storage_bucket"] = storage_bucket
        
        if database_url:
            updates["database_url"] = database_url
        
        return self.update_config(updates)
    
    def setup_service_account(self, key_filename: str) -> bool:
        """Setup service account key filename"""
        return self.update_config({"service_account_key": key_filename})
    
    def enable_local_storage(self, enabled: bool = True) -> bool:
        """Enable or disable local storage"""
        return self.update_config({"local_storage": {"enabled": enabled}})
    
    def setup_collections(self, collections: Dict[str, str]) -> bool:
        """Setup custom collection names"""
        return self.update_config({"collections": collections})
    
    def setup_auto_cleanup(self, enabled: bool, days_to_keep: int = 30, max_files: int = 1000) -> bool:
        """Setup auto cleanup configuration"""
        return self.update_config({
            "auto_cleanup": {
                "enabled": enabled,
                "days_to_keep": days_to_keep,
                "max_files_per_dir": max_files
            }
        })
    
    def create_default_config(self) -> bool:
        """Create default configuration file"""
        return self.save_config(self.config)
    
    def get_config_summary(self) -> str:
        """Get configuration summary"""
        config = self.load_config()
        
        summary = f"""
🔥 FIREBASE CONFIGURATION SUMMARY
=====================================
Project ID: {config.get('project_id', 'Not Set')}
Storage Bucket: {config.get('storage_bucket', 'Not Set')}
Service Account: {config.get('service_account_key', 'Not Set')}
Database URL: {config.get('database_url', 'Not Set')}

Collections:
- Alerts: {config.get('collections', {}).get('alerts', 'detection_alerts')}
- Summaries: {config.get('collections', {}).get('summaries', 'alert_summaries')}
- Status: {config.get('collections', {}).get('status', 'system_status')}
- Evidence: {config.get('collections', {}).get('evidence', 'evidence_files')}

Local Storage: {'Enabled' if config.get('local_storage', {}).get('enabled') else 'Disabled'}
Auto Cleanup: {'Enabled' if config.get('auto_cleanup', {}).get('enabled') else 'Disabled'}

Service Account Available: {'Yes' if self.is_service_account_available() else 'No'}
Config File: {self.config_file}
=====================================
        """
        return summary
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        config = self.load_config()
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check service account
        if not self.is_service_account_available():
            validation["errors"].append("Service account key file not found")
            validation["valid"] = False
        else:
            validation["info"].append("Service account key file found")
        
        # Check project ID
        if not config.get("project_id"):
            validation["errors"].append("Project ID not set")
            validation["valid"] = False
        else:
            validation["info"].append(f"Project ID set: {config['project_id']}")
        
        # Check storage bucket
        if not config.get("storage_bucket"):
            validation["warnings"].append("Storage bucket not set")
        else:
            validation["info"].append(f"Storage bucket set: {config['storage_bucket']}")
        
        # Check local storage
        if config.get("local_storage", {}).get("enabled"):
            validation["info"].append("Local storage enabled")
            
            # Check local directories
            local_config = config.get("local_storage", {})
            for dir_type in ["alerts_dir", "summaries_dir", "status_dir"]:
                dir_path = self.base_dir / local_config.get(dir_type, "")
                if not dir_path.exists():
                    validation["warnings"].append(f"Local directory not found: {dir_path}")
        else:
            validation["info"].append("Local storage disabled")
        
        return validation
    
    def print_validation_results(self):
        """Print validation results"""
        validation = self.validate_config()
        
        print("\n" + "="*60)
        print("🔥 FIREBASE CONFIGURATION VALIDATION")
        print("="*60)
        
        if validation["valid"]:
            print("✅ Configuration is VALID")
        else:
            print("❌ Configuration has ERRORS")
        
        if validation["errors"]:
            print("\n❌ ERRORS:")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        if validation["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        if validation["info"]:
            print("\nℹ️ INFO:")
            for info in validation["info"]:
                print(f"  - {info}")
        
        print("="*60 + "\n")

# Global configuration manager
firebase_config_manager = FirebaseConfigManager()

# Example usage and setup functions
def setup_firebase_basic():
    """Basic Firebase setup"""
    print("🔥 Setting up basic Firebase configuration...")
    
    # Create default config
    firebase_config_manager.create_default_config()
    
    # Print current config
    print(firebase_config_manager.get_config_summary())
    
    # Validate config
    firebase_config_manager.print_validation_results()

def setup_firebase_project(project_id: str, storage_bucket: str = None):
    """Setup Firebase project"""
    print(f"🔥 Setting up Firebase project: {project_id}")
    
    if storage_bucket:
        success = firebase_config_manager.setup_firebase_project(project_id, storage_bucket)
        print(f"✓ Project setup: {project_id} (Bucket: {storage_bucket})")
    else:
        success = firebase_config_manager.setup_firebase_project(project_id)
        print(f"✓ Project setup: {project_id}")
    
    return success

def setup_service_account(key_filename: str = "serviceAccountKey.json"):
    """Setup service account key"""
    print(f"🔥 Setting up service account: {key_filename}")
    
    success = firebase_config_manager.setup_service_account(key_filename)
    
    if success:
        print(f"✓ Service account configured: {key_filename}")
    else:
        print(f"❌ Failed to configure service account")
    
    return success

def enable_firebase_local_storage(enabled: bool = True):
    """Enable/disable local storage"""
    status = "enabled" if enabled else "disabled"
    print(f"🔥 Local storage {status}")
    
    success = firebase_config_manager.enable_local_storage(enabled)
    
    if success:
        print(f"✓ Local storage {status}")
    else:
        print(f"❌ Failed to {status} local storage")
    
    return success

if __name__ == "__main__":
    # Example setup
    setup_firebase_basic()
    
    # Example project setup
    # setup_firebase_project("my-project-id", "my-project.appspot.com")
    
    # Example service account setup
    # setup_service_account("my-service-account.json")
    
    # Example local storage setup
    # enable_firebase_local_storage(True)
