import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from datetime import datetime
import random
from typing import List, Dict, Any
from collections import defaultdict, deque
import math

def load_model():
    """Load YOLOv8 model"""
    print("Loading YOLOv8 model...")
    try:
        return YOLO("yolov8n.pt")
    except:
        print("Downloading YOLOv8 model...")
        return YOLO("yolov8n.yaml")

def initialize_tracker():
    """Initialize DeepSort tracker with enhanced occlusion handling"""
    return DeepSort(
        max_age=100,        # Increased from 50 to 100 frames (~3 seconds at 30fps)
        n_init=3,           # Reduced from 5 to 3 for faster tracking
        max_iou_distance=0.7,  # IoU threshold for matching
        nn_budget=100,      # Feature memory budget
        embedder="mobilenet",  # Use mobilenet embedder
        embedder_gpu=False   # Use CPU to avoid numpy conflicts
    )

def load_pose_model():
    """Load YOLOv8 pose model"""
    print("Loading YOLOv8 Pose Model...")
    try:
        return YOLO("yolov8s-pose.pt")
    except:
        print("Downloading YOLOv8 Pose Model...")
        return YOLO("yolov8s-pose.yaml")

# ----------------------------
# Activity Classifier
# ----------------------------
class ActivityClassifier:
    def __init__(self):
        self.pose_history = defaultdict(lambda: deque(maxlen=15))
        self.activity_history = defaultdict(lambda: deque(maxlen=10))
        self.frame_count = defaultdict(int)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def calculate_body_height(self, keypoints):
        """Calculate vertical body height"""
        if len(keypoints) < 17:
            return 0
        
        # Use shoulder-hip distance as body height indicator
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Average shoulder and hip positions
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        return abs(shoulder_y - hip_y)

    def calculate_motion(self, track_id, keypoints):
        """Calculate frame-to-frame motion"""
        if len(self.pose_history[track_id]) < 2:
            return 0
        
        prev_kp = self.pose_history[track_id][-2]
        curr_kp = keypoints
        
        # Use ankle displacement for motion calculation
        if len(prev_kp) >= 17 and len(curr_kp) >= 17:
            left_ankle_prev = prev_kp[15]
            left_ankle_curr = curr_kp[15]
            right_ankle_prev = prev_kp[16]
            right_ankle_curr = curr_kp[16]
            
            # Calculate displacement
            left_disp = np.linalg.norm(left_ankle_curr - left_ankle_prev)
            right_disp = np.linalg.norm(right_ankle_curr - right_ankle_prev)
            
            return (left_disp + right_disp) / 2
        
        return 0

    def check_body_orientation(self, keypoints):
        """Check if body is vertical or horizontal"""
        if len(keypoints) < 17:
            return "unknown"
        
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Average positions
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # Check if shoulders are above hips (vertical)
        if shoulder_y < hip_y - 20:
            return "vertical"
        elif abs(shoulder_y - hip_y) < 30:
            return "horizontal"
        else:
            return "diagonal"

    def calculate_velocity(self, track_id, keypoints):
        """Calculate movement velocity for walking/running detection"""
        if len(self.pose_history[track_id]) < 2:
            return 0
        
        prev_kp = self.pose_history[track_id][-2]
        curr_kp = keypoints
        
        if len(prev_kp) >= 17 and len(curr_kp) >= 17:
            # Calculate center of mass (hip center)
            prev_hip_center = ((prev_kp[11][0] + prev_kp[12][0]) / 2, 
                               (prev_kp[11][1] + prev_kp[12][1]) / 2)
            curr_hip_center = ((curr_kp[11][0] + curr_kp[12][0]) / 2, 
                               (curr_kp[11][1] + curr_kp[12][1]) / 2)
            
            # Calculate velocity (pixels per frame)
            dx = curr_hip_center[0] - prev_hip_center[0]
            dy = curr_hip_center[1] - prev_hip_center[1]
            velocity = np.sqrt(dx**2 + dy**2)
            
            return velocity
        
        return 0

    def classify_single_frame(self, track_id, keypoints):
        """Professional activity classification with velocity and posture analysis"""
        if len(keypoints) < 17:
            return "Unknown"
        
        # Extract key points
        left_hip = keypoints[11]
        left_knee = keypoints[13]
        left_ankle = keypoints[15]
        right_hip = keypoints[12]
        right_knee = keypoints[14]
        right_ankle = keypoints[16]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # Check visibility of keypoints
        def is_visible(kp):
            return len(kp) >= 2 and kp[0] > 0 and kp[1] > 0
        
        # Count visible keypoints
        shoulder_visible = is_visible(left_shoulder) and is_visible(right_shoulder)
        hip_visible = is_visible(left_hip) and is_visible(right_hip)
        knee_visible = is_visible(left_knee) and is_visible(right_knee)
        ankle_visible = is_visible(left_ankle) and is_visible(right_ankle)
        
        # Calculate key metrics if keypoints are visible
        avg_knee_angle = 180  # Default straight legs
        body_height = 0
        velocity = 0
        
        if knee_visible:
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        if shoulder_visible and hip_visible:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            body_height = abs(shoulder_y - hip_y)
        
        # Calculate velocity for motion analysis
        velocity = self.calculate_velocity(track_id, keypoints)
        
        # ENHANCED SITTING DETECTION LOGIC
        
        # Case 1: Only shoulders visible (person sitting, lower body hidden)
        if shoulder_visible and not hip_visible and not knee_visible:
            # Check for HandsUp pose - arms raised above head/shoulders
            left_elbow = keypoints[7] if len(keypoints) > 7 else None
            right_elbow = keypoints[8] if len(keypoints) > 8 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            left_shoulder = keypoints[5] if len(keypoints) > 5 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 else None
            
            # Check if arms are visible and raised
            arms_visible = (is_visible(left_elbow) and is_visible(left_wrist) and 
                          is_visible(right_elbow) and is_visible(right_wrist))
            
            if arms_visible and is_visible(left_shoulder) and is_visible(right_shoulder):
                # Check if wrists are above shoulders (hands up)
                left_shoulder_y = left_shoulder[1]
                right_shoulder_y = right_shoulder[1]
                left_wrist_y = left_wrist[1]
                right_wrist_y = right_wrist[1]
                
                # Hands up if both wrists are above shoulders
                if left_wrist_y < left_shoulder_y - 20 and right_wrist_y < right_shoulder_y - 20:
                    print("DEBUG: HandsUp pose detected - Arms above shoulders")
                    return "HandsUp"
                
                # Check for Aiming pose - one or both arms extended forward like holding weapon
                # Calculate arm extension angles (shoulder to wrist direction)
                def calculate_arm_angle(shoulder, wrist):
                    dx = wrist[0] - shoulder[0]
                    dy = wrist[1] - shoulder[1]
                    return np.degrees(np.arctan2(dy, dx))
                
                left_arm_angle = calculate_arm_angle(left_shoulder, left_wrist)
                right_arm_angle = calculate_arm_angle(right_shoulder, right_wrist)
                
                # Aiming pose: arms extended forward (horizontal or slightly downward)
                # Forward direction: angle between -45° and +45° (0° = straight forward)
                # NOTE: Camera mirror effect - actual right arm appears as left in mirror
                # So we swap the logic for real-world aiming detection
                left_aiming = -45 <= left_arm_angle <= 45
                right_aiming = -45 <= right_arm_angle <= 45
                
                # Enhanced Aiming Detection: Check if aiming pose is detected
                # Priority: Aiming > Everything else
                if left_aiming or right_aiming:
                    # Check wrist positions relative to shoulders for enhanced detection
                    left_shoulder_y = left_shoulder[1]
                    right_shoulder_y = right_shoulder[1]
                    left_wrist_y = left_wrist[1]
                    right_wrist_y = right_wrist[1]
                    
                    # Enhanced Weapon Detection with better angle ranges
                    # Forward direction: angle between -60° and +60° (0° = straight forward)
                    # NOTE: Camera mirror effect - actual right arm appears as left in mirror
                    # So we swap the logic for real-world aiming detection
                    
                    # Check which arm is actually aiming in real world
                    real_left_aiming = right_aiming  # Mirror: right arm appears as left
                    real_right_aiming = left_aiming   # Mirror: left arm appears as right
                    
                    if real_left_aiming and real_right_aiming:
                        print(f"DEBUG: Dual-Handed Weapon Detected (Sitting) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL DUAL]")
                        return "Aiming"
                    elif real_left_aiming:
                        # Real left arm aiming (appears as right arm in mirror)
                        print(f"DEBUG: Left-Handed Weapon Detected (Sitting) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL LEFT]")
                        return "Aiming"
                    elif real_right_aiming:
                        # Real right arm aiming (appears as left arm in mirror)
                        print(f"DEBUG: Right-Handed Weapon Detected (Sitting) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL RIGHT]")
                        return "Aiming"
                else:
                    # No aiming detected - check for normal activities
                    print(f"DEBUG: No Weapon Detected - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [NORMAL POSITION]")
                    # Continue to other checks (HandsUp, Sitting, etc.)
            
            # If not hands up or aiming, then sitting (default case 1)
            return "Sitting"
        
        # Case 2: Shoulders + hips visible (upper body only, sitting position)
        if shoulder_visible and hip_visible and not knee_visible:
            # Check for HandsUp pose first (priority over sitting/standing)
            left_elbow = keypoints[7] if len(keypoints) > 7 else None
            right_elbow = keypoints[8] if len(keypoints) > 8 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            left_shoulder = keypoints[5] if len(keypoints) > 5 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 else None
            
            # Check if arms are visible and raised
            arms_visible = (is_visible(left_elbow) and is_visible(left_wrist) and 
                          is_visible(right_elbow) and is_visible(right_wrist))
            
            if arms_visible and is_visible(left_shoulder) and is_visible(right_shoulder):
                # Check if wrists are above shoulders (hands up)
                left_shoulder_y = left_shoulder[1]
                right_shoulder_y = right_shoulder[1]
                left_wrist_y = left_wrist[1]
                right_wrist_y = right_wrist[1]
                
                # Hands up if both wrists are above shoulders
                if left_wrist_y < left_shoulder_y - 20 and right_wrist_y < right_shoulder_y - 20:
                    print("DEBUG: HandsUp pose detected (Standing) - Arms above shoulders")
                    return "HandsUp"
                
                # Check for Aiming pose - one or both arms extended forward like holding weapon
                # Calculate arm extension angles (shoulder to wrist direction)
                def calculate_arm_angle(shoulder, wrist):
                    dx = wrist[0] - shoulder[0]
                    dy = wrist[1] - shoulder[1]
                    return np.degrees(np.arctan2(dy, dx))
                
                left_arm_angle = calculate_arm_angle(left_shoulder, left_wrist)
                right_arm_angle = calculate_arm_angle(right_shoulder, right_wrist)
                
                # Aiming pose: arms extended forward (horizontal or slightly downward)
                # Forward direction: angle between -45° and +45° (0° = straight forward)
                # NOTE: Camera mirror effect - actual right arm appears as left in mirror
                # So we swap the logic for real-world aiming detection
                left_aiming = -45 <= left_arm_angle <= 45
                right_aiming = -45 <= right_arm_angle <= 45
                
                # Enhanced Aiming Detection: Check if aiming pose is detected
                # Priority: Aiming > Everything else
                if left_aiming or right_aiming:
                    # Check wrist positions relative to shoulders for enhanced detection
                    left_shoulder_y = left_shoulder[1]
                    right_shoulder_y = right_shoulder[1]
                    left_wrist_y = left_wrist[1]
                    right_wrist_y = right_wrist[1]
                    
                    # Enhanced Weapon Detection with better angle ranges
                    # Forward direction: angle between -60° and +60° (0° = straight forward)
                    # NOTE: Camera mirror effect - actual right arm appears as left in mirror
                    # So we swap the logic for real-world aiming detection
                    
                    # Check which arm is actually aiming in real world
                    real_left_aiming = right_aiming  # Mirror: right arm appears as left
                    real_right_aiming = left_aiming   # Mirror: left arm appears as right
                    
                    if real_left_aiming and real_right_aiming:
                        print(f"DEBUG: Dual-Handed Weapon Detected (Standing) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL DUAL]")
                        return "Aiming"
                    elif real_left_aiming:
                        # Real left arm aiming (appears as right arm in mirror)
                        print(f"DEBUG: Left-Handed Weapon Detected (Standing) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL LEFT]")
                        return "Aiming"
                    elif real_right_aiming:
                        # Real right arm aiming (appears as left arm in mirror)
                        print(f"DEBUG: Right-Handed Weapon Detected (Standing) - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [WEAPON AIMING - REAL RIGHT]")
                        return "Aiming"
                else:
                    # No aiming detected - check for normal activities
                    print(f"DEBUG: No Weapon Detected - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [NORMAL POSITION]")
                    # Continue to other checks (HandsUp, Sitting, etc.)
                
                # Check for HandsUp pose ONLY if no aiming detected
                # Hands up if both wrists are above shoulders (strictly vertical position)
                left_shoulder_y = left_shoulder[1]
                right_shoulder_y = right_shoulder[1]
                left_wrist_y = left_wrist[1]
                right_wrist_y = right_wrist[1]
                
                # Additional check: hands should be significantly above shoulders AND more vertical than horizontal
                left_vertical_distance = left_shoulder_y - left_wrist_y
                right_vertical_distance = right_shoulder_y - right_wrist_y
                
                # Hands up only if: 1) Above shoulders AND 2) More vertical than horizontal
                hands_up_vertical = (left_wrist_y < left_shoulder_y - 30 and right_wrist_y < right_shoulder_y - 30 and
                                   left_vertical_distance > abs(left_wrist[0] - left_shoulder[0]) and
                                   right_vertical_distance > abs(right_wrist[0] - right_shoulder[0]))
                
                if hands_up_vertical:
                    print("DEBUG: HandsUp pose detected - Arms above shoulders (vertical)")
                    return "HandsUp"
            
            # If not hands up or aiming, then check sitting/standing based on body height
            if body_height < 80:  # Increased threshold for sitting
                return "Sitting"
            else:
                return "Standing"
        
        # Case 3: Shoulders + hips + knees visible (your specific requirement)
        if shoulder_visible and hip_visible and knee_visible and not ankle_visible:
            # Check if knees are straight (standing) or bent (sitting)
            # Debug: Print angle for troubleshooting
            print(f"DEBUG: Knee angle = {avg_knee_angle:.1f}°")
            
            # INVERTED LOGIC: External angle < 60° means straight knees (standing)
            # External angle > 60° means bent knees (sitting)
            if avg_knee_angle < 60:  # Straight knees (external angle)
                return "Standing"
            else:  # Bent knees
                return "Sitting"
        
        # Case 4: Walking detection (your specific requirements)
        # Sub-case 4a: Shoulders + Hips + Arms + Knees movement
        if shoulder_visible and hip_visible and knee_visible:
            # Extract arm keypoints
            left_elbow = keypoints[7] if len(keypoints) > 7 else None
            right_elbow = keypoints[8] if len(keypoints) > 8 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            
            # Check if arms are visible
            arms_visible = (is_visible(left_elbow) and is_visible(left_wrist) and 
                          is_visible(right_elbow) and is_visible(right_wrist))
            
            # Calculate velocity for movement detection
            velocity = self.calculate_velocity(track_id, keypoints)
            
            # Your requirement 1: Shoulders + Hips + Arms visible + Knees moving
            # Distinguish between walking and running based on velocity
            if shoulder_visible and hip_visible and arms_visible:
                if 2 <= velocity <= 20:  # Normal walking speed
                    print(f"DEBUG: Walking detected (Shoulders+Hips+Arms+Knees) - Arms: {arms_visible}, Velocity: {velocity:.1f}")
                    return "Walking"
                elif velocity > 20:  # Fast movement = running
                    print(f"DEBUG: Running detected (Shoulders+Hips+Arms+Knees) - Arms: {arms_visible}, Velocity: {velocity:.1f}")
                    return "Running"
        
        # Sub-case 4b: Shoulders + Hips + Arms + Knees + Ankles movement
        if shoulder_visible and hip_visible and knee_visible and ankle_visible:
            # Extract arm keypoints
            left_elbow = keypoints[7] if len(keypoints) > 7 else None
            right_elbow = keypoints[8] if len(keypoints) > 8 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            
            # Check if arms are visible
            arms_visible = (is_visible(left_elbow) and is_visible(left_wrist) and 
                          is_visible(right_elbow) and is_visible(right_wrist))
            
            # Calculate velocity for movement detection
            velocity = self.calculate_velocity(track_id, keypoints)
            
            # Your requirement 2: Shoulders + Hips + Arms + Knees + Ankles moving
            # Distinguish between walking and running based on velocity
            if shoulder_visible and hip_visible and arms_visible:
                if 2 <= velocity <= 20:  # Normal walking speed
                    print(f"DEBUG: Walking detected (Shoulders+Hips+Arms+Knees+Ankles) - Arms: {arms_visible}, Velocity: {velocity:.1f}")
                    return "Walking"
                elif velocity > 20:  # Fast movement = running
                    print(f"DEBUG: Running detected (Shoulders+Hips+Arms+Knees+Ankles) - Arms: {arms_visible}, Velocity: {velocity:.1f}")
                    return "Running"
        
        # Case 5: Full body visible - use professional logic
        if shoulder_visible and hip_visible and knee_visible and ankle_visible:
            # Check for HandsUp pose first (priority over all other activities)
            left_elbow = keypoints[7] if len(keypoints) > 7 else None
            right_elbow = keypoints[8] if len(keypoints) > 8 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            left_shoulder = keypoints[5] if len(keypoints) > 5 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 else None
            
            # Check if arms are visible and raised
            arms_visible = (is_visible(left_elbow) and is_visible(left_wrist) and 
                          is_visible(right_elbow) and is_visible(right_wrist))
            
            if arms_visible and is_visible(left_shoulder) and is_visible(right_shoulder):
                # Calculate arm extension angles (shoulder to wrist direction)
                def calculate_arm_angle(shoulder, wrist):
                    dx = wrist[0] - shoulder[0]
                    dy = wrist[1] - shoulder[1]
                    return np.degrees(np.arctan2(dy, dx))
                
                left_arm_angle = calculate_arm_angle(left_shoulder, left_wrist)
                right_arm_angle = calculate_arm_angle(right_shoulder, right_wrist)
                
                # SIMPLIFIED AIMING DETECTION
                # Forward aiming: arms extended forward (horizontal direction)
                left_aiming = -45 <= left_arm_angle <= 45
                right_aiming = -45 <= right_arm_angle <= 45
                
                # Check for Aiming pose FIRST (highest priority)
                if left_aiming or right_aiming:
                    # Verify arm extension (wrists should be away from shoulders)
                    left_extension = abs(left_wrist[0] - left_shoulder[0])
                    right_extension = abs(right_wrist[0] - right_shoulder[0])
                    body_width = abs(right_shoulder[0] - left_shoulder[0])
                    min_extension = body_width * 0.25  # 25% of shoulder width
                    
                    valid_aiming = False
                    if left_aiming and left_extension > min_extension:
                        valid_aiming = True
                        print(f"DEBUG: Left Arm Aiming - Angle: {left_arm_angle:.1f}°, Extension: {left_extension:.1f}px [AIMING]")
                    if right_aiming and right_extension > min_extension:
                        valid_aiming = True
                        print(f"DEBUG: Right Arm Aiming - Angle: {right_arm_angle:.1f}°, Extension: {right_extension:.1f}px [AIMING]")
                    
                    if valid_aiming:
                        return "Aiming"
                
                # Check for HandsUp pose (second priority)
                # Hands up: wrists above shoulders
                left_hands_up = left_wrist[1] < left_shoulder[1] - 20  # 20 pixels above shoulder
                right_hands_up = right_wrist[1] < right_shoulder[1] - 20
                
                if left_hands_up and right_hands_up:
                    print(f"DEBUG: Hands Up Pose - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [HANDS UP]")
                    return "HandsUp"
                elif left_hands_up or right_hands_up:
                    print(f"DEBUG: Single Hand Up - Left: {left_arm_angle:.1f}°, Right: {right_arm_angle:.1f}° [SINGLE HAND UP]")
                    return "HandsUp"
                
                # Check for HandsUp pose ONLY if no aiming detected
                # Hands up if both wrists are above shoulders (strictly vertical position)
                left_shoulder_y = left_shoulder[1]
                right_shoulder_y = right_shoulder[1]
                left_wrist_y = left_wrist[1]
                right_wrist_y = right_wrist[1]
                
                # Additional check: hands should be significantly above shoulders AND more vertical than horizontal
                left_vertical_distance = left_shoulder_y - left_wrist_y
                right_vertical_distance = right_shoulder_y - right_wrist_y
                
                # Hands up only if: 1) Above shoulders AND 2) More vertical than horizontal
                hands_up_vertical = (left_wrist_y < left_shoulder_y - 30 and right_wrist_y < right_shoulder_y - 30 and
                                   left_vertical_distance > abs(left_wrist[0] - left_shoulder[0]) and
                                   right_vertical_distance > abs(right_wrist[0] - right_shoulder[0]))
                
                if hands_up_vertical:
                    print("DEBUG: HandsUp pose detected - Arms above shoulders (vertical)")
                    return "HandsUp"
            
            # If not hands up or aiming, then use professional logic for other activities
            
            # 1. LYING - Body horizontal, very low height
            if body_height < 35:
                return "Lying"
            
            # 2. SITTING - Bent knees AND compressed body
            # Must have both: knee angle < 110° AND body compressed
            if avg_knee_angle < 110 and body_height < 60:
                return "Sitting"
            
            # 3. STANDING - Straight legs AND low velocity AND upright body
            # Must have: knee angle > 160° AND velocity < 3 AND body upright
            if avg_knee_angle > 160 and velocity < 3 and body_height > 50:
                return "Standing"
            
            # 4. WALKING - Moderate knee bend AND moderate velocity AND upright body
            # Must have: knee angle 130-160° AND velocity 3-8 AND body upright
            if 130 <= avg_knee_angle <= 160 and 3 <= velocity <= 8 and body_height > 50:
                return "Walking"
            
            # 5. RUNNING - High knee bend AND high velocity AND upright body
            # Must have: knee angle ≤ 130° AND velocity > 8 AND body upright
            if avg_knee_angle <= 130 and velocity > 8 and body_height > 50:
                return "Running"
        
        return "Analyzing"

    def classify(self, track_id, keypoints):
        """Classify activity with majority vote for stability"""
        self.pose_history[track_id].append(keypoints)
        self.frame_count[track_id] += 1
        
        # Need at least 3 frames to start classification (reduced from 5)
        if len(self.pose_history[track_id]) < 3:
            return "Analyzing"
        
        # Get current frame classification
        current_activity = self.classify_single_frame(track_id, keypoints)
        
        # Add to history
        self.activity_history[track_id].append(current_activity)
        
        # Use majority vote from last 5 frames (reduced from 10)
        if len(self.activity_history[track_id]) >= 2:
            activities = list(self.activity_history[track_id])
            
            # Count occurrences
            activity_counts = {}
            for activity in activities:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            # Get most common activity (excluding "Analyzing")
            best_activity = "Analyzing"
            max_count = 0
            
            for activity, count in activity_counts.items():
                if activity != "Analyzing" and count > max_count:
                    best_activity = activity
                    max_count = count
            
            # Only return stable activity if it appears in at least 50% of recent frames
            if max_count >= len(activities) * 0.5:
                return best_activity
        
        return current_activity

def get_color_for_id(track_id):
    """Generate consistent random color for each track ID"""
    random.seed(track_id)  # Seed with track ID for consistent colors
    color = (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )
    return color

def draw_pose_landmarks(frame, keypoints, color=(0, 255, 0), thickness=2):
    """Draw pose landmarks and skeleton on frame"""
    if len(keypoints) < 17:
        return frame
    
    # Define pose connections (skeleton)
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Shoulders to hips
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if len(kp) >= 2:
            x, y = int(kp[0]), int(kp[1])
            # Different colors for different body parts
            if i in [0, 1, 2, 3, 4]:  # Face
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)  # Blue
            elif i in [5, 6, 7, 8, 9, 10]:  # Arms
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green
            elif i in [11, 12, 13, 14, 15, 16]:  # Legs
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Red
            else:
                cv2.circle(frame, (x, y), 4, color, -1)
    
    # Draw skeleton connections
    for connection in POSE_CONNECTIONS:
        if connection[0] < len(keypoints) and connection[1] < len(keypoints):
            kp1 = keypoints[connection[0]]
            kp2 = keypoints[connection[1]]
            
            if len(kp1) >= 2 and len(kp2) >= 2:
                x1, y1 = int(kp1[0]), int(kp1[1])
                x2, y2 = int(kp2[0]), int(kp2[1])
                
                # Check if keypoints are visible (confidence > 0)
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame

def draw_activity_info(frame, track_id, activity, keypoints, bbox):
    """Draw detailed activity information with angles"""
    x1, y1, x2, y2 = bbox
    
    if len(keypoints) >= 17:
        # Calculate angles for display
        left_hip = keypoints[11]
        left_knee = keypoints[13]
        left_ankle = keypoints[15]
        right_hip = keypoints[12]
        right_knee = keypoints[14]
        right_ankle = keypoints[16]
        
        # Calculate knee angles
        activity_classifier = ActivityClassifier()
        left_knee_angle = activity_classifier.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = activity_classifier.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Draw angle information
        info_y = y1 - 40
        
        # Activity label
        cv2.putText(frame, f"Activity: {activity}", (x1, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Knee angles
        cv2.putText(frame, f"L Knee: {left_knee_angle:.1f}°", (x1, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        cv2.putText(frame, f"R Knee: {right_knee_angle:.1f}°", (x1, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        cv2.putText(frame, f"Avg: {avg_knee_angle:.1f}°", (x1, info_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
        # Draw angle arcs on the person
        if len(left_knee) >= 2 and len(left_hip) >= 2 and len(left_ankle) >= 2:
            # Draw left knee angle
            center = (int(left_knee[0]), int(left_knee[1]))
            cv2.circle(frame, center, 20, (255, 200, 0), 1)
        
        if len(right_knee) >= 2 and len(right_hip) >= 2 and len(right_ankle) >= 2:
            # Draw right knee angle
            center = (int(right_knee[0]), int(right_knee[1]))
            cv2.circle(frame, center, 20, (255, 200, 0), 1)
    
    return frame

def get_detections(frame, model, pose_model, confidence_threshold=0.35):
    """Get person detections from frame with pose data"""
    results = model(frame, conf=confidence_threshold)
    pose_results = pose_model(frame, conf=0.4)
    
    detections = []
    pose_data = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Only detect people (class 0)
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter small detections (likely false positives)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Minimum area threshold (adjust based on camera distance)
                    min_area = 1000  # pixels
                    
                    if area > min_area:
                        detections.append(([x1, y1, width, height], conf, 'person'))
    
    # Extract pose keypoints
    for pose_result in pose_results:
        if pose_result.keypoints is not None:
            keypoints = pose_result.keypoints.xy.cpu().numpy()
            if keypoints.shape[0] > 0: 
                pose_data.append(keypoints[0])
            else:
                pose_data.append(None)
        else:
            pose_data.append(None)
            
    return detections, pose_data

def draw_occlusion_status(frame, tracks):
    """Draw occlusion status for tracks"""
    h, w = frame.shape[:2]
    y_offset = 130
    
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            
            # Check if track is recently updated
            time_since_update = track.time_since_update if hasattr(track, 'time_since_update') else 0
            
            if time_since_update > 0:
                # Draw occlusion warning
                status_text = f"Person {track_id}: Occluded ({time_since_update} frames)"
                color = (0, 0, 255) if time_since_update > 30 else (0, 165, 255)  # Red if long occlusion
                
                cv2.putText(frame, status_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
    
    return frame

def draw_info(frame, current_count, unique_count, tracks):
    """Draw information on frame"""
    h, w = frame.shape[:2]
    
    # Draw background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Draw text
    cv2.putText(frame, f"Current People: {current_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Unique: {unique_count}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
    """Get person detections from frame with occlusion handling"""
    results = model(frame, conf=confidence_threshold)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Only detect people (class 0)
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter small detections (likely false positives)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Minimum area threshold (adjust based on camera distance)
                    min_area = 1000  # pixels
                    
                    if area > min_area:
                        detections.append(([x1, y1, width, height], conf, 'person'))
    
    return detections

class HumanTracker:
    """Human Tracker using DeepSort algorithm with advanced activity classification"""
    
    def __init__(self, model_path="yolov8n.pt", pose_model_path="yolov8s-pose.pt", camera_index=0):
        """
        Initialize Human Tracker with DeepSort and Activity Classification
        
        Args:
            model_path: Path to YOLOv8 model
            pose_model_path: Path to YOLOv8 pose model
            camera_index: Camera index (default 0 for webcam)
        """
        # Load YOLOv8 models
        self.model = load_model()
        self.pose_model = load_pose_model()
        print(" Models loaded")
        
        # Initialize DeepSort tracker
        self.tracker = initialize_tracker()
        print(" DeepSort tracker initialized")
        
        # Initialize activity classifier
        self.activity_classifier = ActivityClassifier()
        print(" Activity classifier initialized")
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        
        # Tracking parameters
        self.frame_count = 0
        self.confidence_threshold = 0.35
        
        # Colors for visualization (consistent colors per ID)
        self.id_colors = {}  # Persistent colors per ID
    
    def detect_humans(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect humans using DeepSort tracking with advanced activity classification
        
        Args:
            frame: Input frame
            
        Returns:
            List of human detection dictionaries with IDs and activity data
        """
        # Update frame count
        self.frame_count += 1
        
        # Get detections and pose data
        detections, pose_data = get_detections(frame, self.model, self.pose_model, self.confidence_threshold)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Convert tracks to detection format with activity classification
        final_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            
            # Get track confidence
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            confidence = float(confidence) if confidence is not None else 0.5
            
            # Match pose data to track for activity classification
            activity = "Unknown"
            keypoints = None
            if pose_data:
                # Find closest pose to this track
                track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                min_dist = float('inf')
                selected_pose = None
                
                for pose in pose_data:
                    if pose is not None and len(pose) > 16:   # Ensure we have enough keypoints
                        pose_center = (pose[0][0], pose[0][1])  # Use nose as center
                        dist = np.sqrt((pose_center[0] - track_center[0])**2 + (pose_center[1] - track_center[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            selected_pose = pose
                
                if selected_pose is not None:
                    keypoints = selected_pose
                    activity = self.activity_classifier.classify(track_id, selected_pose)
            
            detection = {
                "id": track_id,
                "bbox": [x1, y1, w, h],
                "person_conf": confidence,
                "gun_conf": 0.0,
                "knife_conf": 0.0,
                "fight_conf": 0.0,
                "meta": {
                    "class_name": "PERSON",
                    "raw_confidence": confidence,
                    "frame": self.frame_count,
                    "camera": self.camera_index,
                    "time_since_update": track.time_since_update if hasattr(track, 'time_since_update') else 0,
                    "is_occluded": track.time_since_update > 0 if hasattr(track, 'time_since_update') else False,
                    "activity": activity,
                    "keypoints": keypoints if keypoints is not None else []
                },
                "timestamp": time.time(),
                "frame": frame.copy()
            }
            
            final_detections.append(detection)
        
        return final_detections
    
    def get_id_color(self, person_id: int) -> tuple:
        """Get consistent color for person ID"""
        if person_id not in self.id_colors:
            # Generate consistent color based on ID
            self.id_colors[person_id] = get_color_for_id(person_id)
        
        return self.id_colors[person_id]
    
    def update_frame_count(self):
        """Update frame counter"""
        self.frame_count += 1
    
    def draw_tracking_info(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw tracking information on frame with advanced activity classification
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with tracking information drawn
        """
        annotated = frame.copy()
        
        # Draw information overlay
        current_count = len(detections)
        unique_ids = set([d["id"] for d in detections])
        unique_count = len(unique_ids)
        
        # Draw background for text
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        # Draw text
        cv2.putText(annotated, f"Current People: {current_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Total Unique: {unique_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw bounding boxes and IDs for each detection with activity info
        for detection in detections:
            track_id = detection["id"]
            x1, y1, w, h = detection["bbox"]
            x2, y2 = x1 + w, y1 + h
            
            # Get unique color for this person
            color = self.get_id_color(track_id)
            
            # Check if track is occluded
            is_occluded = detection["meta"].get("is_occluded", False)
            time_since_update = detection["meta"].get("time_since_update", 0)
            activity = detection["meta"].get("activity", "Unknown")
            keypoints = detection["meta"].get("keypoints", [])
            
            # Draw bounding box with unique color
            if is_occluded:
                # Draw dashed bounding box for occluded person
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    start_x = min(i, x2)
                    end_x = min(i + dash_length, x2)
                    cv2.line(annotated, (start_x, y1), (end_x, y1), color, 3)
                    cv2.line(annotated, (start_x, y2), (end_x, y2), color, 3)
                
                for i in range(y1, y2, dash_length * 2):
                    start_y = min(i, y2)
                    end_y = min(i + dash_length, y2)
                    cv2.line(annotated, (x1, start_y), (x1, end_y), color, 3)
                    cv2.line(annotated, (x2, start_y), (x2, end_y), color, 3)
            else:
                # Solid bounding box for visible person
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw ID and activity label
            if is_occluded:
                label = f"Person {track_id} - OCCLUDED ({time_since_update}f) - {activity}"
                # Red tint for occluded
                label_color = (0, 0, 255)
            else:
                label = f"Person {track_id} - {activity}"
                label_color = color
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw colored background for label
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), label_color, -1)
            
            # Draw text in white for contrast
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw pose landmarks if available
            if len(keypoints) >= 17:
                annotated = draw_pose_landmarks(annotated, keypoints, color, 2)
                annotated = draw_activity_info(annotated, track_id, activity, keypoints, (x1, y1, x2, y2))
            
            # Draw center point (different style for occluded)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if is_occluded:
                # Draw X for occluded person
                cv2.drawMarker(annotated, (center_x, center_y), label_color, 
                             cv2.MARKER_CROSS, 10, 3)
            else:
                # Draw filled circle for visible person
                cv2.circle(annotated, (center_x, center_y), 5, color, -1)
                cv2.circle(annotated, (center_x, center_y), 7, (255, 255, 255), 1)
        
        return annotated

def main():
    """Main function to run camera tracking with advanced activity classification"""
    print("🎯 Starting People Tracking & Activity Recognition System...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize tracker
    tracker = HumanTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully")
    print("📹 Starting live feed...")
    
    # Tracking variables
    unique_people = set()
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Detect and track humans with activity classification
        detections = tracker.detect_humans(frame)
        
        # Draw tracking information with activity
        frame = tracker.draw_tracking_info(frame, detections)
        
        # Update unique people set
        for detection in detections:
            unique_people.add(detection["id"])
        
        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow("People Tracking & Activity Recognition System", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Quitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"activity_tracking_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Session Summary:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Total Unique People: {len(unique_people)}")
    print(f"   Session Duration: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
