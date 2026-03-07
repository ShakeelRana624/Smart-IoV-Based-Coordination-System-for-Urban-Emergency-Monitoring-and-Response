"""
Firebase Setup Script for Intelligent Weapon Detection System

Easy setup and configuration of Firebase for alert storage.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.firebase_config import firebase_config_manager

def print_banner():
    """Print setup banner"""
    print("\n" + "="*80)
    print("🔥 FIREBASE SETUP WIZARD")
    print("🔥 Intelligent Weapon Detection System")
    print("="*80)
    print("This wizard will help you configure Firebase for alert storage.")
    print("You can choose between Firebase cloud storage or local JSON storage.")
    print("="*80 + "\n")

def check_service_account():
    """Check if service account key exists"""
    service_account_path = firebase_config_manager.get_service_account_path()
    
    if os.path.exists(service_account_path):
        print(f"✅ Service account key found: {service_account_path}")
        return True
    else:
        print(f"❌ Service account key not found: {service_account_path}")
        print("   Please download your service account key from Firebase Console")
        print("   and place it in the project root directory.")
        return False

def setup_firebase_cloud():
    """Setup Firebase cloud storage"""
    print("\n🔥 Setting up Firebase Cloud Storage...")
    
    # Check service account
    if not check_service_account():
        print("❌ Cannot proceed with Firebase setup without service account key.")
        return False
    
    # Get project details
    project_id = input("Enter your Firebase Project ID (default: weapon-detection-system): ").strip()
    if not project_id:
        project_id = "weapon-detection-system"
    
    storage_bucket = input("Enter your Storage Bucket (default: weapon-detection-system.appspot.com): ").strip()
    if not storage_bucket:
        storage_bucket = "weapon-detection-system.appspot.com"
    
    database_url = input("Enter your Database URL (optional): ").strip()
    
    # Setup configuration
    success = firebase_config_manager.setup_firebase_project(project_id, storage_bucket, database_url)
    
    if success:
        print(f"✅ Firebase project configured: {project_id}")
        print(f"✅ Storage bucket: {storage_bucket}")
        if database_url:
            print(f"✅ Database URL: {database_url}")
        
        # Test connection
        print("\n🔥 Testing Firebase connection...")
        try:
            from utils.firebase_alert_storage import FirebaseAlertStorage
            storage = FirebaseAlertStorage()
            
            if storage.firebase_available:
                print("✅ Firebase connection successful!")
                return True
            else:
                print("❌ Firebase connection failed. Using local storage.")
                return False
        except Exception as e:
            print(f"❌ Firebase test failed: {e}")
            return False
    else:
        print("❌ Failed to configure Firebase project")
        return False

def setup_local_storage():
    """Setup local JSON storage"""
    print("\n📁 Setting up Local JSON Storage...")
    
    # Get local storage preferences
    alerts_dir = input("Alerts directory (default: firebase_alerts): ").strip()
    if not alerts_dir:
        alerts_dir = "firebase_alerts"
    
    summaries_dir = input("Summaries directory (default: firebase_summaries): ").strip()
    if not summaries_dir:
        summaries_dir = "firebase_summaries"
    
    status_dir = input("Status directory (default: firebase_status): ").strip()
    if not status_dir:
        status_dir = "firebase_status"
    
    evidence_dir = input("Evidence directory (default: firebase_evidence): ").strip()
    if not evidence_dir:
        evidence_dir = "firebase_evidence"
    
    # Setup local storage configuration
    collections_config = {
        "alerts_dir": alerts_dir,
        "summaries_dir": summaries_dir,
        "status_dir": status_dir,
        "evidence_dir": evidence_dir
    }
    
    success = firebase_config_manager.update_config({"local_storage": collections_config})
    
    if success:
        print("✅ Local storage configuration updated")
        
        # Create directories
        for dir_name, dir_path in collections_config.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Directory created: {dir_path}")
        
        return True
    else:
        print("❌ Failed to configure local storage")
        return False

def setup_auto_cleanup():
    """Setup automatic cleanup"""
    print("\n🧹 Setting up Automatic Cleanup...")
    
    enable_cleanup = input("Enable automatic cleanup? (y/n, default: y): ").strip().lower()
    enabled = enable_cleanup != 'n'
    
    if enabled:
        days_to_keep = input("Days to keep files (default: 30): ").strip()
        try:
            days_to_keep = int(days_to_keep) if days_to_keep else 30
        except ValueError:
            days_to_keep = 30
        
        max_files = input("Maximum files per directory (default: 1000): ").strip()
        try:
            max_files = int(max_files) if max_files else 1000
        except ValueError:
            max_files = 1000
        
        success = firebase_config_manager.setup_auto_cleanup(enabled, days_to_keep, max_files)
        
        if success:
            print(f"✅ Auto cleanup enabled")
            print(f"✅ Files older than {days_to_keep} days will be deleted")
            print(f"✅ Maximum {max_files} files per directory")
        else:
            print("❌ Failed to configure auto cleanup")
    else:
        firebase_config_manager.setup_auto_cleanup(False)
        print("✅ Auto cleanup disabled")

def setup_collections():
    """Setup custom collection names"""
    print("\n📚 Setting up Collection Names...")
    
    use_custom = input("Use custom collection names? (y/n, default: n): ").strip().lower()
    
    if use_custom == 'y':
        alerts_collection = input("Alerts collection (default: detection_alerts): ").strip()
        summaries_collection = input("Summaries collection (default: alert_summaries): ").strip()
        status_collection = input("Status collection (default: system_status): ").strip()
        evidence_collection = input("Evidence collection (default: evidence_files): ").strip()
        
        collections = {
            "alerts": alerts_collection if alerts_collection else "detection_alerts",
            "summaries": summaries_collection if summaries_collection else "alert_summaries",
            "status": status_collection if status_collection else "system_status",
            "evidence": evidence_collection if evidence_collection else "evidence_files"
        }
        
        success = firebase_config_manager.setup_collections(collections)
        
        if success:
            print("✅ Custom collection names configured")
            for name, collection in collections.items():
                print(f"✅ {name.title()}: {collection}")
        else:
            print("❌ Failed to configure custom collections")
    else:
        print("✅ Using default collection names")

def main():
    """Main setup function"""
    print_banner()
    
    # Show current configuration
    print("📋 Current Configuration:")
    print(firebase_config_manager.get_config_summary())
    
    # Setup choices
    print("\n🔥 Setup Options:")
    print("1. Firebase Cloud Storage")
    print("2. Local JSON Storage Only")
    print("3. Configure Collections")
    print("4. Configure Auto Cleanup")
    print("5. Show Current Configuration")
    print("6. Validate Configuration")
    print("7. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            setup_firebase_cloud()
        elif choice == '2':
            setup_local_storage()
        elif choice == '3':
            setup_collections()
        elif choice == '4':
            setup_auto_cleanup()
        elif choice == '5':
            print("\n📋 Current Configuration:")
            print(firebase_config_manager.get_config_summary())
        elif choice == '6':
            firebase_config_manager.print_validation_results()
        elif choice == '7':
            print("\n👋 Setup completed!")
            print("🔥 Firebase configuration is ready.")
            print("📁 Run the main system to start storing alerts.")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()
