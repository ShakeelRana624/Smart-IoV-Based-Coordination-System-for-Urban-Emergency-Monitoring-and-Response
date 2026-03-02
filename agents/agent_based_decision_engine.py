"""
Agent-Based Hybrid Decision Engine with LangGraph

Enhanced decision engine using LangGraph agents for:
- Multi-agent threat assessment
- Coordinated response actions
- Adaptive learning and memory
- Parallel processing capabilities
"""

import time
import asyncio
import threading
import cv2
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import json
from collections import deque

# LangGraph imports
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE_CHECKPOINT = True
except ImportError:
    HAS_SQLITE_CHECKPOINT = False
    print("Warning: SQLite checkpoint not available, using memory checkpoint")
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Import existing engine
from .hybrid_decision_engine import DecisionEngine, Threat, State

# --------------------------
# System State Management
# --------------------------
from enum import Enum

class SystemState(Enum):
    """System states for threat progression"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    THREAT_DETECTION = "threat_detection"
    EMERGENCY = "emergency"
    
    def __str__(self):
        return self.value

class StateTransition:
    """Manages system state transitions"""
    
    def __init__(self):
        self.current_state = SystemState.NORMAL
        self.state_history = []
        self.state_durations = {}
        self.last_state_change = time.time()
        self.suspicious_threshold = 0.3  # Low confidence triggers suspicious
        self.threat_threshold = 0.4     # Medium confidence triggers threat detection
        self.emergency_threshold = 0.7  # High confidence triggers emergency
        
    def update_state(self, detection: Dict[str, Any]) -> tuple[SystemState, bool]:
        """
        Update system state based on detection
        
        Returns:
            tuple: (new_state, state_changed)
        """
        old_state = self.current_state
        new_state = self._determine_new_state(detection)
        
        # Record state change
        if new_state != old_state:
            self._record_state_change(old_state, new_state)
            self.current_state = new_state
            return new_state, True
        
        return old_state, False
    
    def _determine_new_state(self, detection: Dict[str, Any]) -> SystemState:
        """Determine new state based on detection"""
        # Get maximum weapon confidence
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        explosion_conf = detection.get("explosion_conf", 0)
        grenade_conf = detection.get("grenade_conf", 0)
        
        max_confidence = max(gun_conf, knife_conf, explosion_conf, grenade_conf)
        
        # Check for suspicious behavior
        is_suspicious = self._is_suspicious_behavior(detection)
        
        # State transition logic
        if max_confidence >= self.emergency_threshold:
            return SystemState.EMERGENCY
        elif max_confidence >= self.threat_threshold:
            return SystemState.THREAT_DETECTION
        elif max_confidence >= self.suspicious_threshold or is_suspicious:
            return SystemState.SUSPICIOUS
        else:
            return SystemState.NORMAL
    
    def _is_suspicious_behavior(self, detection: Dict[str, Any]) -> bool:
        """Check for suspicious behavior patterns"""
        meta = detection.get("meta", {})
        
        # Suspicious indicators
        suspicious_indicators = [
            meta.get("running", False),
            meta.get("loitering", False),
            meta.get("concealment", False),
            meta.get("erratic_movement", False)
        ]
        
        return any(suspicious_indicators)
    
    def _record_state_change(self, old_state: SystemState, new_state: SystemState):
        """Record state change for tracking"""
        current_time = time.time()
        duration = current_time - self.last_state_change
        
        # Update state duration
        if old_state not in self.state_durations:
            self.state_durations[old_state] = []
        self.state_durations[old_state].append(duration)
        
        # Add to history
        self.state_history.append({
            "from_state": old_state,
            "to_state": new_state,
            "timestamp": current_time,
            "duration": duration
        })
        
        self.last_state_change = current_time
        
        print(f"🔄 State Transition: {old_state.value} → {new_state.value} (Duration: {duration:.2f}s)")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        total_time = sum(sum(durations) for durations in self.state_durations.values())
        
        return {
            "current_state": self.current_state.value,
            "total_transitions": len(self.state_history),
            "state_durations": {
                state.value: {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "avg_duration": sum(durations) / len(durations) if durations else 0
                }
                for state, durations in self.state_durations.items()
            },
            "time_in_current_state": time.time() - self.last_state_change,
            "total_session_time": total_time
        }

class EmergencyManager:
    """Manages emergency state operations"""
    
    def __init__(self):
        self.emergency_active = False
        self.emergency_start_time = None
        self.emergency_actions = []
        self.uav_dispatched = False
        self.authorities_notified = False
        
    def activate_emergency(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Activate emergency state"""
        self.emergency_active = True
        self.emergency_start_time = time.time()
        
        emergency_response = {
            "emergency_level": "CRITICAL",
            "activation_time": self.emergency_start_time,
            "detection_id": detection.get("id"),
            "threat_type": self._identify_threat_type(detection),
            "immediate_actions": self._initiate_emergency_actions(detection),
            "coordination_required": self._determine_coordination_needs(detection)
        }
        
        print(f"🚨 EMERGENCY ACTIVATED: {emergency_response['threat_type']}")
        return emergency_response
    
    def _identify_threat_type(self, detection: Dict[str, Any]) -> str:
        """Identify type of threat"""
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        explosion_conf = detection.get("explosion_conf", 0)
        grenade_conf = detection.get("grenade_conf", 0)
        
        if gun_conf > 0.5:
            return "FIREARM_THREAT"
        elif explosion_conf > 0.5:
            return "EXPLOSIVE_THREAT"
        elif grenade_conf > 0.5:
            return "GRENADE_THREAT"
        elif knife_conf > 0.5:
            return "WEAPON_THREAT"
        else:
            return "UNKNOWN_THREAT"
    
    def _initiate_emergency_actions(self, detection: Dict[str, Any]) -> List[str]:
        """Initiate emergency response actions"""
        actions = []
        
        # UAV dispatch
        if not self.uav_dispatched:
            actions.append("DISPATCH_UAV")
            self.uav_dispatched = True
        
        # Authorities notification
        if not self.authorities_notified:
            actions.append("NOTIFY_AUTHORITIES")
            self.authorities_notified = True
        
        # Immediate alerts
        actions.extend([
            "ACTIVATE_ALL_ALARMS",
            "LOCKDOWN_FACILITY",
            "EMERGENCY_BROADCAST",
            "EVACUATION_INITIATE"
        ])
        
        self.emergency_actions.extend(actions)
        return actions
    
    def _determine_coordination_needs(self, detection: Dict[str, Any]) -> List[str]:
        """Determine coordination requirements"""
        needs = []
        
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        
        if gun_conf > 0.5:
            needs.extend(["SWAT_TEAM", "NEGOTIATOR"])
        
        if knife_conf > 0.5:
            needs.extend(["SECURITY_TEAM", "MEDICAL_STANDBY"])
        
        needs.extend(["LAW_ENFORCEMENT", "EMERGENCY_SERVICES"])
        
        return needs
    
    def deactivate_emergency(self) -> Dict[str, Any]:
        """Deactivate emergency state"""
        if not self.emergency_active:
            return {"status": "no_emergency"}
        
        duration = time.time() - self.emergency_start_time if self.emergency_start_time else 0
        
        summary = {
            "emergency_deactivated": True,
            "duration": duration,
            "actions_taken": self.emergency_actions,
            "uav_dispatched": self.uav_dispatched,
            "authorities_notified": self.authorities_notified
        }
        
        # Reset emergency state
        self.emergency_active = False
        self.emergency_start_time = None
        self.emergency_actions = []
        self.uav_dispatched = False
        self.authorities_notified = False
        
        print(f"✅ Emergency deactivated after {duration:.2f} seconds")
        return summary
class AgentState(TypedDict):
    """Shared state for all agents in the workflow"""
    detection: Dict[str, Any]
    system_state: SystemState
    state_changed: bool
    threat_assessment: Dict[str, Any]
    decision: Dict[str, Any]
    evidence: Dict[str, Any]
    notifications: List[Dict[str, Any]]
    memory_context: Dict[str, Any]
    emergency_response: Optional[Dict[str, Any]]
    timestamp: float
    agent_messages: Annotated[List[str], "Messages from agents"]

@dataclass
class ThreatContext:
    """Context for threat assessment"""
    severity: float
    confidence: float
    duration_frames: int
    bayes_prob: float
    historical_patterns: List[str]
    environmental_factors: Dict[str, Any]

# --------------------------
# Agent Implementations
# --------------------------
class PerceptionAgent:
    """Processes and validates detection data from perception models"""
    
    def __init__(self):
        self.validation_rules = {
            "min_confidence": 0.3,
            "max_bbox_area": 50000,
            "min_bbox_area": 100
        }
    
    def process(self, state: AgentState) -> AgentState:
        """Validate and preprocess detection data"""
        detection = state["detection"]
        
        # Validate detection quality
        validated_detection = self._validate_detection(detection)
        
        # Enhance with metadata
        enhanced_detection = self._enhance_detection(validated_detection)
        
        state["detection"] = enhanced_detection
        state["agent_messages"].append("PerceptionAgent: Detection validated and enhanced")
        
        return state
    
    def _validate_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detection data quality"""
        validated = detection.copy()
        
        # Check confidence thresholds
        for key in ["person_conf", "gun_conf", "knife_conf", "fight_conf"]:
            if key in validated and validated[key] is not None and validated[key] < self.validation_rules["min_confidence"]:
                validated[key] = 0.0
        
        # Validate bounding box
        if "bbox" in validated and validated["bbox"]:
            x, y, w, h = validated["bbox"]
            area = w * h
            if area < self.validation_rules["min_bbox_area"] or area > self.validation_rules["max_bbox_area"]:
                validated["bbox_valid"] = False
            else:
                validated["bbox_valid"] = True
        else:
            validated["bbox_valid"] = False
        
        return validated
    
    def _enhance_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Add enhancement metadata"""
        enhanced = detection.copy()
        enhanced["processing_timestamp"] = time.time()
        enhanced["perception_quality"] = self._calculate_quality_score(detection)
        return enhanced
    
    def _calculate_quality_score(self, detection: Dict[str, Any]) -> float:
        """Calculate overall detection quality score"""
        score = 0.0
        factors = 0
        
        # Person detection confidence
        if "person_conf" in detection:
            score += detection["person_conf"]
            factors += 1
        
        # Bounding box validity
        if detection.get("bbox_valid", False):
            score += 0.5
            factors += 1
        
        # Overall confidence
        max_conf = max([
            detection.get("gun_conf", 0),
            detection.get("knife_conf", 0),
            detection.get("fight_conf", 0)
        ])
        if max_conf > 0:
            score += max_conf
            factors += 1
        
        return score / max(factors, 1)

class ThreatAssessmentAgent:
    """Advanced threat assessment using multiple analysis techniques"""
    
    def __init__(self):
        self.pattern_library = {
            "suspicious_behavior": ["loitering", " erratic_movement", "concealment"],
            "weapon_preparation": ["reaching", "drawing", "handling"],
            "violent_intent": ["aggressive_posture", "rapid_movement", "targeting"]
        }
    
    def process(self, state: AgentState) -> AgentState:
        """Perform comprehensive threat assessment"""
        detection = state["detection"]
        
        # Create threat context
        context = self._create_threat_context(detection, state.get("memory_context", {}))
        
        # Multi-dimensional analysis
        threat_assessment = {
            "immediate_threat": self._assess_immediate_threat(context),
            "behavioral_analysis": self._analyze_behavior(context),
            "risk_prediction": self._predict_risk(context),
            "confidence_score": self._calculate_confidence(context),
            "recommended_actions": self._recommend_actions(context)
        }
        
        state["threat_assessment"] = threat_assessment
        state["agent_messages"].append("ThreatAssessmentAgent: Comprehensive threat analysis completed")
        
        return state
    
    def _create_threat_context(self, detection: Dict[str, Any], memory: Dict[str, Any]) -> ThreatContext:
        """Create comprehensive threat context"""
        return ThreatContext(
            severity=self._extract_severity(detection),
            confidence=detection.get("person_conf", 0.0),
            duration_frames=memory.get("duration_frames", 0),
            bayes_prob=memory.get("bayes_prob", 0.0),
            historical_patterns=memory.get("patterns", []),
            environmental_factors=detection.get("environment", {})
        )
    
    def _extract_severity(self, detection: Dict[str, Any]) -> float:
        """Extract immediate severity from detection with pose enhancement"""
        base_severity = 0.0
        
        # Weapon-based severity
        if detection.get("fight_conf", 0) > 0.5:
            base_severity = Threat.CRITICAL
        elif detection.get("gun_conf", 0) > 0.3:
            base_severity = Threat.HIGH
        elif detection.get("knife_conf", 0) > 0.3:
            base_severity = Threat.MEDIUM
        
        # ENHANCEMENT: Add pose-based severity
        activity = detection.get("meta", {}).get("activity", "Unknown")
        pose_severity_multiplier = self._get_pose_severity_multiplier(activity)
        
        # Combine weapon and pose severity
        enhanced_severity = min(base_severity * pose_severity_multiplier, 4.0)
        
        return enhanced_severity
    
    def _get_pose_severity_multiplier(self, activity: str) -> float:
        """Get severity multiplier based on detected activity"""
        pose_severity = {
            "aiming": 1.8,        # Very high threat
            "hands_up": 1.5,      # High threat  
            "running": 1.2,       # Moderate threat
            "walking": 1.0,       # Normal
            "sitting": 0.7,        # Low threat
            "standing": 0.6,      # Very low threat
            "lying": 0.5,          # Minimal threat
            "unknown": 1.0         # Default
        }
        return pose_severity.get(activity.lower(), 1.0)
    
    def _assess_immediate_threat(self, context: ThreatContext) -> Dict[str, Any]:
        """Assess immediate threat level"""
        return {
            "level": context.severity,
            "confidence": context.confidence,
            "factors": self._identify_threat_factors(context),
            "urgency": self._calculate_urgency(context)
        }
    
    def _analyze_behavior(self, context: ThreatContext) -> Dict[str, Any]:
        """Analyze behavioral patterns"""
        return {
            "patterns_detected": self._match_patterns(context),
            "anomaly_score": self._calculate_anomaly(context),
            "intent_inference": self._infer_intent(context)
        }
    
    def _predict_risk(self, context: ThreatContext) -> Dict[str, Any]:
        """Predict future risk escalation"""
        return {
            "escalation_probability": self._calculate_escalation_risk(context),
            "time_to_critical": self._estimate_time_to_critical(context),
            "recommended_monitoring_level": self._recommend_monitoring(context)
        }
    
    def _calculate_confidence(self, context: ThreatContext) -> float:
        """Calculate overall assessment confidence"""
        base_confidence = context.confidence
        temporal_confidence = min(context.duration_frames / 10.0, 1.0)
        bayesian_confidence = context.bayes_prob
        
        return (base_confidence * 0.4 + temporal_confidence * 0.3 + bayesian_confidence * 0.3)
    
    def _recommend_actions(self, context: ThreatContext) -> List[str]:
        """Recommend specific actions based on assessment"""
        actions = []
        
        if context.severity >= Threat.HIGH:
            actions.extend(["IMMEDIATE_ALERT", "EVIDENCE_COLLECTION", "UAV_DISPATCH"])
        elif context.severity >= Threat.MEDIUM:
            actions.extend(["ENHANCED_MONITORING", "EVIDENCE_COLLECTION"])
        elif context.severity >= Threat.LOW:
            actions.append("INCREASED_ATTENTION")
        
        return actions
    
    # Helper methods for threat assessment
    def _identify_threat_factors(self, context: ThreatContext) -> List[str]:
        factors = []
        if context.severity >= Threat.HIGH:
            factors.append("weapon_detected")
        if context.confidence > 0.8:
            factors.append("high_confidence")
        if context.duration_frames > 5:
            factors.append("persistent_threat")
        return factors
    
    def _calculate_urgency(self, context: ThreatContext) -> str:
        if context.severity >= Threat.CRITICAL:
            return "immediate"
        elif context.severity >= Threat.HIGH:
            return "high"
        elif context.severity >= Threat.MEDIUM:
            return "medium"
        return "low"
    
    def _match_patterns(self, context: ThreatContext) -> List[str]:
        # Pattern matching logic would go here
        return context.historical_patterns
    
    def _calculate_anomaly(self, context: ThreatContext) -> float:
        # Anomaly detection logic would go here
        return 0.5  # Placeholder
    
    def _infer_intent(self, context: ThreatContext) -> str:
        if context.severity >= Threat.HIGH:
            return "hostile"
        elif context.severity >= Threat.MEDIUM:
            return "suspicious"
        return "neutral"
    
    def _calculate_escalation_risk(self, context: ThreatContext) -> float:
        base_risk = context.severity / 4.0
        duration_factor = min(context.duration_frames / 20.0, 1.0)
        return min(base_risk + duration_factor * 0.3, 1.0)
    
    def _estimate_time_to_critical(self, context: ThreatContext) -> int:
        if context.severity >= Threat.HIGH:
            return 5  # seconds
        elif context.severity >= Threat.MEDIUM:
            return 15
        return 30
    
    def _recommend_monitoring(self, context: ThreatContext) -> str:
        if context.severity >= Threat.HIGH:
            return "continuous"
        elif context.severity >= Threat.MEDIUM:
            return "frequent"
        return "periodic"

class DecisionCoordinatorAgent:
    """Coordinates final decisions based on all agent inputs"""
    
    def __init__(self):
        self.decision_weights = {
            "threat_assessment": 0.4,
            "historical_context": 0.2,
            "environmental_factors": 0.15,
            "resource_availability": 0.15,
            "operational_constraints": 0.1
        }
    
    def process(self, state: AgentState) -> AgentState:
        """Make final coordinated decision"""
        threat_assessment = state.get("threat_assessment", {})
        memory_context = state.get("memory_context", {})
        
        # Synthesize decision
        decision = self._synthesize_decision(threat_assessment, memory_context, state["detection"])
        
        # Validate decision
        validated_decision = self._validate_decision(decision)
        
        state["decision"] = validated_decision
        state["agent_messages"].append("DecisionCoordinatorAgent: Final decision synthesized")
        
        return state
    
    def _synthesize_decision(self, threat_assessment: Dict[str, Any], 
                           memory_context: Dict[str, Any], 
                           detection: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final decision from all inputs with pose enhancement"""
        
        # Calculate composite threat score
        composite_score = self._calculate_composite_score(threat_assessment, memory_context)
        
        # ENHANCEMENT: Add pose-based decision factors
        activity = detection.get("meta", {}).get("activity", "Unknown")
        pose_decision_factor = self._get_pose_decision_factor(activity)
        
        # Adjust composite score with pose factor
        enhanced_composite_score = min(composite_score * pose_decision_factor, 4.0)
        
        # Determine response level
        response_level = self._determine_response_level(enhanced_composite_score)
        
        # Select actions
        actions = self._select_actions(response_level, threat_assessment, detection)
        
        # ENHANCEMENT: Add pose-specific actions
        if activity in ["aiming", "hands_up"] and "SAVE_EVIDENCE" not in actions:
            actions.append("SAVE_EVIDENCE")
        
        return {
            "composite_score": enhanced_composite_score,
            "response_level": response_level,
            "actions": actions,
            "rationale": self._generate_rationale(threat_assessment, enhanced_composite_score),
            "priority": self._assign_priority(response_level),
            "estimated_duration": self._estimate_action_duration(actions),
            "pose_context": activity  # ENHANCEMENT: Add pose context
        }
    
    def _get_pose_decision_factor(self, activity: str) -> float:
        """Get decision factor based on detected activity"""
        pose_factors = {
            "aiming": 1.5,        # High priority
            "hands_up": 1.3,      # Medium-high priority
            "running": 1.1,       # Slight priority
            "walking": 1.0,       # Normal
            "sitting": 0.8,       # Lower priority
            "standing": 0.7,      # Low priority
            "lying": 0.6,          # Minimal priority
            "unknown": 1.0         # Default
        }
        return pose_factors.get(activity.lower(), 1.0)
    
    def _calculate_composite_score(self, threat_assessment: Dict[str, Any], 
                                 memory_context: Dict[str, Any]) -> float:
        """Calculate composite threat score"""
        immediate_score = threat_assessment.get("immediate_threat", {}).get("level", 0)
        confidence_score = threat_assessment.get("confidence_score", 0)
        risk_score = threat_assessment.get("risk_prediction", {}).get("escalation_probability", 0)
        
        # Weighted combination
        composite = (immediate_score * 0.4 + confidence_score * 4 * 0.3 + risk_score * 4 * 0.3)
        return min(composite, 4.0)
    
    def _determine_response_level(self, composite_score: float) -> str:
        """Determine response level based on composite score"""
        if composite_score >= 3.2:
            return "CRITICAL"
        elif composite_score >= 2.4:
            return "HIGH"
        elif composite_score >= 1.6:
            return "MEDIUM"
        elif composite_score >= 0.8:
            return "LOW"
        return "MINIMAL"
    
    def _select_actions(self, response_level: str, threat_assessment: Dict[str, Any], 
                      detection: Dict[str, Any]) -> List[str]:
        """Select appropriate actions based on response level"""
        actions = []
        
        if response_level == "CRITICAL":
            actions = ["DISPATCH_UAV", "IMMEDIATE_ALERT", "EVIDENCE_COLLECTION", "NOTIFY_ALL"]
        elif response_level == "HIGH":
            actions = ["LOCAL_ALARM", "EVIDENCE_COLLECTION", "NOTIFY_OPERATOR"]
        elif response_level == "MEDIUM":
            actions = ["ENHANCED_MONITORING", "EVIDENCE_COLLECTION"]
        elif response_level == "LOW":
            actions = ["LOG_ONLY"]
        
        # Add recommended actions from threat assessment
        recommended = threat_assessment.get("recommended_actions", [])
        actions.extend([a for a in recommended if a not in actions])
        
        return actions
    
    def _validate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate decision consistency and feasibility"""
        validated = decision.copy()
        
        # Ensure actions are consistent with response level
        response_level = decision.get("response_level", "MINIMAL")
        actions = decision.get("actions", [])
        
        # Validate action coherence
        if response_level == "CRITICAL" and "DISPATCH_UAV" not in actions:
            actions.append("DISPATCH_UAV")
        
        validated["actions"] = actions
        validated["validation_timestamp"] = time.time()
        
        return validated
    
    def _generate_rationale(self, threat_assessment: Dict[str, Any], composite_score: float) -> str:
        """Generate decision rationale"""
        immediate = threat_assessment.get("immediate_threat", {})
        risk = threat_assessment.get("risk_prediction", {})
        
        rationale_parts = [
            f"Composite score: {composite_score:.2f}",
            f"Immediate threat level: {immediate.get('level', 'unknown')}",
            f"Escalation risk: {risk.get('escalation_probability', 0):.2f}"
        ]
        
        return "; ".join(rationale_parts)
    
    def _assign_priority(self, response_level: str) -> int:
        """Assign numerical priority (1=highest)"""
        priorities = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4,
            "MINIMAL": 5
        }
        return priorities.get(response_level, 5)
    
    def _estimate_action_duration(self, actions: List[str]) -> int:
        """Estimate time to complete actions (seconds)"""
        if not actions:
            return 1
        
        duration_map = {
            "DISPATCH_UAV": 120,
            "IMMEDIATE_ALERT": 5,
            "EVIDENCE_COLLECTION": 30,
            "LOCAL_ALARM": 10,
            "ENHANCED_MONITORING": 60,
            "LOG_ONLY": 1,
            "NO_ACTION": 1
        }
        
        return max([duration_map.get(action, 10) for action in actions])

class EvidenceAgent:
    """Intelligent video evidence collection agent with frame buffering"""
    
    def __init__(self, buffer_size: int = 30, fps: int = 30):  # Changed fps to 30 for normal speed
        self.buffer_size = buffer_size  # Number of frames to buffer before detection
        self.fps = fps  # Normal speed recording
        self.frame_buffer = deque(maxlen=buffer_size)
        self.video_writer = None
        self.current_recording = None
        self.recording_start_time = None
        self.storage_path = "evidence/videos/"
        self.last_detection_time = 0
        self.recording_lock = threading.Lock()
        
        # Store current detection and result for annotations
        self.current_detection = None
        self.current_result = None
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Recording states - FIXED: Only one recording at a time
        self.is_recording = False
        self.weapon_detected = False
        self.normal_state_counter = 0
        self.normal_state_threshold = 100  # ENHANCED: 100 frames (3.3 seconds) after weapon disappears
        self.recording_completed = False  # Track if recording is completed for this session
    
    def add_frame_to_buffer(self, frame: np.ndarray, timestamp: float = None):
        """Add frame to circular buffer for pre-detection evidence"""
        if timestamp is None:
            timestamp = time.time()
        
        frame_data = {
            'frame': frame.copy(),
            'timestamp': timestamp
        }
        self.frame_buffer.append(frame_data)
    
    def process(self, state: AgentState) -> AgentState:
        """Process evidence collection with intelligent video capture - FIXED LOGIC"""
        decision = state.get("decision", {})
        detection = state["detection"]
        current_frame = detection.get("frame")
        
        # Store current detection and result for annotations
        self.current_detection = detection
        self.current_result = {
            "system_state": state.get("system_state", "normal"),
            "state_changed": state.get("state_changed", False),
            "threat_score": decision.get("composite_score", 0),
            "action": "|".join(decision.get("actions", ["NO_ACTION"])),
            "emergency_response": state.get("emergency_response")
        }
        
        if current_frame is not None:
            # Add current frame to buffer
            self.add_frame_to_buffer(current_frame, detection.get("timestamp"))
        
        # Check if weapon detected
        weapon_detected = self._is_weapon_detected(detection)
        
        # FIXED LOGIC: Only record when weapon is detected, and only one recording
        if weapon_detected and not self.is_recording and not self.recording_completed:
            # Start recording with buffered frames
            self._start_recording(detection)
            self.weapon_detected = True
            self.last_detection_time = time.time()
            print(f"🎯 Weapon detected! Started recording evidence")
        
        elif self.is_recording:
            if weapon_detected:
                # Weapon still detected, continue recording
                self._continue_recording(current_frame, detection, self.current_result)
                self.last_detection_time = time.time()
                self.normal_state_counter = 0
            else:
                # No weapon detected, check if we should stop
                self.normal_state_counter += 1
                if self.normal_state_counter >= self.normal_state_threshold:
                    self._stop_recording()
                    self.recording_completed = True  # Mark as completed to prevent new recordings
                    print(f"✅ Weapon no longer detected. Recording completed.")
                else:
                    self._continue_recording(current_frame, detection, self.current_result)
        
        evidence_plan = self._plan_evidence_collection(decision, detection)
        
        state["evidence"] = evidence_plan
        state["agent_messages"].append(f"EvidenceAgent: {'Recording' if self.is_recording else 'Monitoring'} evidence")
        
        return state
    
    def _is_weapon_detected(self, detection: Dict[str, Any]) -> bool:
        """Check if any weapon is detected with sufficient confidence OR aiming pose detected"""
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        explosion_conf = detection.get("explosion_conf", 0)  # If available
        grenade_conf = detection.get("grenade_conf", 0)  # If available
        
        # Check for aiming pose activity
        activity = detection.get("meta", {}).get("activity", "Unknown")
        is_aiming = activity.lower() == "aiming"
        
        # Enhanced weapon detection: Weapon OR (Weapon + Aiming)
        weapon_detected = max(gun_conf, knife_conf, explosion_conf, grenade_conf) > 0.4
        
        # ENHANCEMENT: Save evidence if:
        # 1. Weapon detected (any confidence > 0.4)
        # 2. OR Weapon detected + Aiming pose (even low confidence weapon)
        weapon_plus_aiming = weapon_detected and is_aiming
        
        if weapon_plus_aiming:
            print(f"🎯 Enhanced Evidence: Weapon + Aiming detected! Weapon: {max(gun_conf, knife_conf, explosion_conf, grenade_conf):.2f}, Pose: {activity}")
            return True
        elif weapon_detected:
            print(f"🔫 Evidence: Weapon detected! Confidence: {max(gun_conf, knife_conf, explosion_conf, grenade_conf):.2f}")
            return True
        elif is_aiming:
            print(f"👁️ Evidence: Aiming pose detected! (No weapon)")
            return False  # Only aiming without weapon - don't save evidence
        
        return False
    
    def _start_recording(self, detection: Dict[str, Any]):
        """Start video recording with buffered frames"""
        with self.recording_lock:
            if self.is_recording:
                return
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_id = detection.get("id", "unknown")
            self.current_recording = f"weapon_detection_{detection_id}_{timestamp}.mp4"
            output_path = os.path.join(self.storage_path, self.current_recording)
            
            # Get frame dimensions from buffered frames or current frame
            if self.frame_buffer:
                sample_frame = self.frame_buffer[0]['frame']
            else:
                sample_frame = detection.get("frame")
            
            if sample_frame is not None:
                height, width = sample_frame.shape[:2]
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    output_path, fourcc, self.fps, (width, height)
                )
                
                if self.video_writer.isOpened():
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    
                    # Write buffered frames first (pre-detection evidence)
                    self._write_buffered_frames()
                    
                    print(f"✓ EvidenceAgent: Started recording {self.current_recording}")
                else:
                    print(f"❌ EvidenceAgent: Failed to initialize video writer")
            else:
                print("❌ EvidenceAgent: No frame available for recording")
    
    def _write_buffered_frames(self):
        """Write all buffered frames to video (pre-detection evidence)"""
        if not self.video_writer or not self.frame_buffer:
            return
        
        for frame_data in self.frame_buffer:
            # For buffered frames, we don't have detection results yet
            frame = self._annotate_frame(frame_data['frame'], frame_data['timestamp'], "PRE-DETECTION")
            self.video_writer.write(frame)
        
        print(f"✓ EvidenceAgent: Wrote {len(self.frame_buffer)} buffered frames")
    
    def _continue_recording(self, frame: np.ndarray, detection: Dict[str, Any] = None, result: Dict[str, Any] = None):
        """Continue recording with current frame and annotations"""
        with self.recording_lock:
            if not self.is_recording or self.video_writer is None or frame is None:
                return
            
            # Annotate frame with detection info
            annotated_frame = self._annotate_frame(frame, time.time(), "DETECTION", detection, result)
            self.video_writer.write(annotated_frame)
    
    def _stop_recording(self):
        """Stop recording and save video"""
        with self.recording_lock:
            if not self.is_recording:
                return
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            recording_duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            print(f"✓ EvidenceAgent: Stopped recording {self.current_recording}")
            print(f"  Duration: {recording_duration:.2f} seconds")
            print(f"  Saved to: {os.path.join(self.storage_path, self.current_recording)}")
            
            self.is_recording = False
            self.current_recording = None
            self.recording_start_time = None
            self.weapon_detected = False
            self.normal_state_counter = 0
            # Note: recording_completed is set in the process method to control timing
    
    def _annotate_frame(self, frame: np.ndarray, timestamp: float, status: str, 
                        detection: Dict[str, Any] = None, result: Dict[str, Any] = None) -> np.ndarray:
        """Annotate frame with comprehensive detection information"""
        annotated = frame.copy()
        
        # Add timestamp
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add status
        cv2.putText(annotated, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add recording indicator
        if self.is_recording:
            cv2.putText(annotated, "● RECORDING", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add detection annotations if available
        if detection and result:
            annotated = self._add_detection_annotations(annotated, detection, result)
        
        return annotated
    
    def _add_detection_annotations(self, frame: np.ndarray, detection: Dict[str, Any], result: Dict[str, Any]) -> np.ndarray:
        """Add comprehensive detection annotations to frame"""
        # Get system state and color
        system_state = result.get("system_state", "normal")
        if hasattr(system_state, 'value'):
            system_state_str = system_state.value.upper()
        else:
            system_state_str = str(system_state).upper()
        
        state_color = self._get_state_color(system_state_str)
        
        # Draw detection bounding box
        bbox = detection.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            cv2.rectangle(frame, (x, y), (x + w, y + h), state_color, 3)
            
            # Add detection label
            detection_id = detection.get("id", "Unknown")
            label = f"ID:{detection_id} {system_state_str}"
            threat_score = result.get("threat_score", 0)
            score_text = f"Score:{threat_score:.1f}"
            
            # Background for label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - 50), (x + label_size[0], y), state_color, -1)
            
            # Label text
            cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, score_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        # ENHANCEMENT: Add weapon bounding box when weapon detected
        self._add_weapon_bounding_box(frame, detection)
        
        # Add system state indicator (top-right corner)
        cv2.putText(frame, f"State: {system_state_str}", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Add state change indicator
        if result.get("state_changed", False):
            cv2.putText(frame, "STATE CHANGED!", (frame.shape[1] - 250, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add weapon confidence indicators (left side)
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        explosion_conf = detection.get("explosion_conf", 0)
        grenade_conf = detection.get("grenade_conf", 0)
        
        confidences = [
            ("Gun", gun_conf, (0, 0, 255)),
            ("Knife", knife_conf, (255, 165, 0)),
            ("Explosion", explosion_conf, (255, 0, 0)),
            ("Grenade", grenade_conf, (255, 0, 255))
        ]
        
        y_offset = 120
        for weapon_name, confidence, color in confidences:
            if confidence > 0.1:  # Only show if confidence is significant
                conf_text = f"{weapon_name}: {confidence:.2f}"
                cv2.putText(frame, conf_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
        
        # Add actions being taken (bottom-left)
        actions = result.get("action", "").split("|") if result.get("action") else []
        if actions and actions != [""]:
            action_text = "Actions: " + ", ".join(actions[:3])  # Show first 3 actions
            if len(actions) > 3:
                action_text += "..."
            cv2.putText(frame, action_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add emergency indicator if active
        emergency_response = result.get("emergency_response")
        if emergency_response and not emergency_response.get("emergency_deactivated"):
            threat_type = emergency_response.get("threat_type", "UNKNOWN")
            cv2.putText(frame, f"🚨 EMERGENCY: {threat_type}", 
                       (frame.shape[1] // 2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return frame
    
    def _add_weapon_bounding_box(self, frame: np.ndarray, detection: Dict[str, Any]) -> None:
        """Add weapon bounding box when weapon is detected"""
        # Check if any weapon is detected
        gun_conf = detection.get("gun_conf", 0)
        knife_conf = detection.get("knife_conf", 0)
        explosion_conf = detection.get("explosion_conf", 0)
        grenade_conf = detection.get("grenade_conf", 0)
        
        max_confidence = max(gun_conf, knife_conf, explosion_conf, grenade_conf)
        
        # Only show weapon box if weapon confidence > 0.4
        if max_confidence > 0.4:
            # Get person bounding box
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                
                # Create weapon bounding box (smaller, inside person box)
                # Position weapon box in upper-right area of person box (where weapon is typically held)
                weapon_box_width = w // 3  # 1/3 of person width
                weapon_box_height = h // 4  # 1/4 of person height
                weapon_box_x = x + w - weapon_box_width - 10  # Right side with padding
                weapon_box_y = y + h // 3  # Upper-middle area
                
                # Draw red weapon bounding box
                cv2.rectangle(frame, 
                           (weapon_box_x, weapon_box_y), 
                           (weapon_box_x + weapon_box_width, weapon_box_y + weapon_box_height), 
                           (0, 0, 255), 2)  # Red color
                
                # Add weapon label
                weapon_type = "GUN"
                if knife_conf > gun_conf and knife_conf > max_confidence * 0.8:
                    weapon_type = "KNIFE"
                elif explosion_conf > max_confidence * 0.8:
                    weapon_type = "EXPLOSION"
                elif grenade_conf > max_confidence * 0.8:
                    weapon_type = "GRENADE"
                
                weapon_label = f"🔫 {weapon_type}"
                label_size = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for weapon label
                cv2.rectangle(frame, 
                           (weapon_box_x, weapon_box_y - 20), 
                           (weapon_box_x + label_size[0], weapon_box_y), 
                           (0, 0, 255), -1)  # Red background
                
                # Weapon label text
                cv2.putText(frame, weapon_label, 
                          (weapon_box_x, weapon_box_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add confidence score
                conf_text = f"{max_confidence:.2f}"
                cv2.putText(frame, conf_text, 
                          (weapon_box_x, weapon_box_y + weapon_box_height + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def _get_state_color(self, state: str) -> tuple:
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
    
    def _plan_evidence_collection(self, decision: Dict[str, Any], detection: Dict[str, Any]) -> Dict[str, Any]:
        """Plan evidence collection strategy"""
        actions = decision.get("actions", [])
        
        evidence_plan = {
            "collection_type": "video" if self.is_recording else "standard",
            "recording_active": self.is_recording,
            "video_file": self.current_recording if self.is_recording else None,
            "buffered_frames": len(self.frame_buffer),
            "duration": self._calculate_recording_duration(),
            "quality_level": "high" if decision.get("response_level") == "CRITICAL" else "standard",
            "storage_priority": decision.get("priority", 5),
            "metadata": {
                "detection_id": detection.get("id"),
                "timestamp": detection.get("timestamp"),
                "threat_level": decision.get("response_level"),
                "bbox": detection.get("bbox"),
                "weapon_detected": self._is_weapon_detected(detection)
            }
        }
        
        return evidence_plan
    
    def _calculate_recording_duration(self) -> int:
        """Calculate current recording duration in seconds"""
        if not self.is_recording or not self.recording_start_time:
            return 0
        
        return int(time.time() - self.recording_start_time)
    
    def force_stop_recording(self):
        """Force stop recording (for cleanup)"""
        if self.is_recording:
            self._stop_recording()
    
    def reset_session(self):
        """Reset evidence agent for new recording session"""
        with self.recording_lock:
            self.recording_completed = False
            self.normal_state_counter = 0
            self.weapon_detected = False
            print("🔄 EvidenceAgent: Reset for new session - ready to record next weapon detection")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current evidence agent status"""
        return {
            "is_recording": self.is_recording,
            "current_file": self.current_recording,
            "buffered_frames": len(self.frame_buffer),
            "recording_duration": self._calculate_recording_duration(),
            "weapon_detected": self.weapon_detected,
            "normal_state_counter": self.normal_state_counter,
            "recording_completed": self.recording_completed
        }

class NotificationAgent:
    """Manages notifications and alerts"""
    
    def __init__(self):
        self.notification_channels = {
            "operator": "webhook",
            "uav": "api_call",
            "local": "system_alarm",
            "emergency": "broadcast"
        }
    
    def process(self, state: AgentState) -> AgentState:
        """Process notifications based on decision"""
        decision = state.get("decision", {})
        detection = state["detection"]
        
        notifications = self._generate_notifications(decision, detection)
        
        state["notifications"] = notifications
        state["agent_messages"].append(f"NotificationAgent: {len(notifications)} notifications prepared")
        
        return state
    
    def _generate_notifications(self, decision: Dict[str, Any], detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate appropriate notifications"""
        notifications = []
        actions = decision.get("actions", [])
        response_level = decision.get("response_level", "MINIMAL")
        
        if "NOTIFY_OPERATOR" in actions or "NOTIFY_ALL" in actions:
            notifications.append({
                "channel": "operator",
                "type": "alert",
                "priority": decision.get("priority", 5),
                "message": f"Threat detected: {response_level}",
                "metadata": {
                    "detection_id": detection.get("id"),
                    "bbox": detection.get("bbox"),
                    "timestamp": detection.get("timestamp")
                }
            })
        
        if "DISPATCH_UAV" in actions:
            notifications.append({
                "channel": "uav",
                "type": "dispatch",
                "priority": 1,
                "message": "UAV dispatch required",
                "metadata": {
                    "target_location": detection.get("bbox"),
                    "urgency": "immediate" if response_level == "CRITICAL" else "high"
                }
            })
        
        if "LOCAL_ALARM" in actions:
            notifications.append({
                "channel": "local",
                "type": "alarm",
                "priority": 2,
                "message": "Local security alert",
                "metadata": {}
            })
        
        return notifications

class StateManagementAgent:
    """Manages system state transitions and emergency responses"""
    
    def __init__(self):
        self.state_transition = StateTransition()
        self.emergency_manager = EmergencyManager()
    
    def process(self, state: AgentState) -> AgentState:
        """Process state management"""
        detection = state["detection"]
        
        # Update system state
        new_state, state_changed = self.state_transition.update_state(detection)
        
        # Update state in agent state
        state["system_state"] = new_state
        state["state_changed"] = state_changed
        
        # Handle emergency state
        emergency_response = None
        if new_state == SystemState.EMERGENCY and state_changed:
            emergency_response = self.emergency_manager.activate_emergency(detection)
        elif new_state != SystemState.EMERGENCY and self.emergency_manager.emergency_active:
            emergency_response = self.emergency_manager.deactivate_emergency()
        
        state["emergency_response"] = emergency_response
        
        # Add state information to messages
        state_msg = f"StateManagementAgent: State = {new_state.value}"
        if state_changed:
            state_msg += f" (Changed from {self.state_transition.state_history[-1]['from_state'].value if self.state_transition.state_history else 'Unknown'})"
        
        if emergency_response:
            if emergency_response.get("emergency_deactivated"):
                state_msg += f" | Emergency deactivated"
            else:
                state_msg += f" | Emergency activated: {emergency_response.get('threat_type', 'Unknown')}"
        
        state["agent_messages"].append(state_msg)
        
        return state
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        summary = self.state_transition.get_state_summary()
        summary["emergency_active"] = self.emergency_manager.emergency_active
        return summary

class MemoryAgent:
    """Maintains temporal context and learning"""
    
    def __init__(self):
        self.memory_store = {}
        self.pattern_history = []
    
    def process(self, state: AgentState) -> AgentState:
        """Update memory and provide context"""
        detection = state["detection"]
        threat_assessment = state.get("threat_assessment", {})
        system_state = state.get("system_state", SystemState.NORMAL)
        
        # Update memory with system state
        memory_context = self._update_memory(detection, threat_assessment, system_state)
        
        state["memory_context"] = memory_context
        state["agent_messages"].append("MemoryAgent: Context updated")
        
        return state
    
    def _update_memory(self, detection: Dict[str, Any], threat_assessment: Dict[str, Any], system_state: SystemState) -> Dict[str, Any]:
        """Update memory with new information and pose patterns"""
        detection_id = detection.get("id")
        
        if detection_id not in self.memory_store:
            self.memory_store[detection_id] = {
                "first_seen": detection.get("timestamp"),
                "detection_count": 0,
                "patterns": [],
                "max_threat_level": 0,
                "state_history": [],
                "pose_history": [],  # ENHANCEMENT: Track pose patterns
                "activity_counts": {}  # ENHANCEMENT: Count activity occurrences
            }
        
        memory = self.memory_store[detection_id]
        memory["detection_count"] += 1
        memory["last_seen"] = detection.get("timestamp")
        memory["last_state"] = system_state.value
        
        # ENHANCEMENT: Track pose patterns
        activity = detection.get("meta", {}).get("activity", "Unknown")
        memory["pose_history"].append({
            "activity": activity,
            "timestamp": detection.get("timestamp"),
            "threat_level": threat_assessment.get("immediate_threat", {}).get("level", 0)
        })
        
        # ENHANCEMENT: Count activity occurrences
        if activity not in memory["activity_counts"]:
            memory["activity_counts"][activity] = 0
        memory["activity_counts"][activity] += 1
        
        # Add state to history
        memory["state_history"].append({
            "state": system_state.value,
            "timestamp": detection.get("timestamp"),
            "activity": activity  # ENHANCEMENT: Add activity to state history
        })
        
        # Update max threat level
        current_threat = threat_assessment.get("immediate_threat", {}).get("level", 0)
        memory["max_threat_level"] = max(memory["max_threat_level"], current_threat)
        
        # ENHANCEMENT: Analyze pose patterns
        if len(memory["pose_history"]) >= 5:  # Need at least 5 poses for pattern analysis
            memory["patterns"] = self._analyze_pose_patterns(memory["pose_history"])
        
        # Return enhanced memory context
        return {
            "detection_id": detection_id,
            "detection_count": memory["detection_count"],
            "duration": detection.get("timestamp") - memory["first_seen"],
            "max_threat_level": memory["max_threat_level"],
            "state_history": memory["state_history"][-10:],  # Last 10 states
            "pose_patterns": memory.get("patterns", []),
            "activity_counts": memory["activity_counts"],  # ENHANCEMENT: Activity statistics
            "dominant_activity": max(memory["activity_counts"].items(), key=lambda x: x[1])[0] if memory["activity_counts"] else "unknown"
        }
    
    def _analyze_pose_patterns(self, pose_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze pose patterns for suspicious behavior"""
        patterns = []
        
        # Check for rapid pose changes
        if len(pose_history) >= 3:
            recent_poses = pose_history[-3:]
            activities = [p["activity"] for p in recent_poses]
            
            # Pattern 1: Rapid activity changes
            if len(set(activities)) == 3:  # 3 different activities
                patterns.append({
                    "type": "rapid_activity_change",
                    "severity": "medium",
                    "description": "Rapid pose changes detected",
                    "activities": activities
                })
            
            # Pattern 2: Escalating threat
            threat_levels = [p["threat_level"] for p in recent_poses]
            if threat_levels == sorted(threat_levels) and threat_levels[-1] > 2.0:
                patterns.append({
                    "type": "escalating_threat",
                    "severity": "high",
                    "description": "Threat level escalating",
                    "threat_progression": threat_levels
                })
        
        # Pattern 3: Aiming detection
        aiming_count = sum(1 for p in pose_history if p["activity"] == "aiming")
        if aiming_count >= 2:
            patterns.append({
                "type": "repeated_aiming",
                "severity": "high",
                "description": "Repeated aiming behavior",
                "aiming_frequency": aiming_count / len(pose_history)
            })
        
        return patterns

# --------------------------
# Agent-Based Decision Engine
# --------------------------
class AgentBasedDecisionEngine:
    """Main engine coordinating all agents using LangGraph"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize agents
        self.perception_agent = PerceptionAgent()
        self.state_agent = StateManagementAgent()
        self.threat_agent = ThreatAssessmentAgent()
        self.decision_agent = DecisionCoordinatorAgent()
        self.evidence_agent = EvidenceAgent()
        self.notification_agent = NotificationAgent()
        self.memory_agent = MemoryAgent()
        
        # Keep original engine for compatibility
        self.legacy_engine = DecisionEngine(config)
        
        # Build LangGraph workflow
        self.checkpointer = None
        if HAS_SQLITE_CHECKPOINT:
            self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        else:
            self.checkpointer = None
        self.workflow = self._build_workflow()
        
        # Configuration
        self.config = config or {}
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("perception", self.perception_agent.process)
        workflow.add_node("state_management", self.state_agent.process)
        workflow.add_node("memory_update", self.memory_agent.process)
        workflow.add_node("threat_assessment", self.threat_agent.process)
        workflow.add_node("decision_coordination", self.decision_agent.process)
        workflow.add_node("evidence_planning", self.evidence_agent.process)
        workflow.add_node("notification_planning", self.notification_agent.process)
        
        # Define conditional routing based on system state
        def route_after_perception(state: AgentState) -> str:
            system_state = state.get("system_state", SystemState.NORMAL)
            if system_state == SystemState.EMERGENCY:
                return "decision_coordination"  # Skip to decision for emergency
            else:
                return "state_management"  # Normal flow
        
        def route_after_threat(state: AgentState) -> str:
            system_state = state.get("system_state", SystemState.NORMAL)
            threat_level = state.get("threat_assessment", {}).get("immediate_threat", {}).get("level", 0)
            
            # High threat or emergency state goes to decision
            if threat_level >= Threat.HIGH or system_state == SystemState.EMERGENCY:
                return "decision_coordination"
            else:
                return "decision_coordination"  # Always go to decision for consistency
        
        # Add edges
        workflow.set_entry_point("perception")
        workflow.add_conditional_edges("perception", route_after_perception)
        workflow.add_edge("state_management", "memory_update")
        workflow.add_edge("memory_update", "threat_assessment")
        workflow.add_conditional_edges("threat_assessment", route_after_threat)
        workflow.add_edge("decision_coordination", "evidence_planning")
        workflow.add_edge("evidence_planning", "notification_planning")
        workflow.add_edge("notification_planning", END)
        
        return workflow.compile(checkpointer=self.checkpointer) if self.checkpointer else workflow.compile()
    
    def process(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Process detection through agent workflow"""
        # Initialize state
        initial_state = AgentState(
            detection=detection,
            system_state=SystemState.NORMAL,
            state_changed=False,
            threat_assessment={},
            decision={},
            evidence={},
            notifications=[],
            memory_context={},
            emergency_response=None,
            timestamp=time.time(),
            agent_messages=[]
        )
        
        # Run workflow
        config = RunnableConfig(thread_id=f"detection_{detection.get('id')}")
        result = self.workflow.invoke(initial_state, config)
        
        # Execute actions (integrate with legacy engine)
        self._execute_actions(result)
        
        # Return formatted result compatible with original engine
        return self._format_result(result)
    
    def _execute_actions(self, state: AgentState):
        """Execute decided actions"""
        notifications = state.get("notifications", [])
        evidence = state.get("evidence", {})
        
        # Execute notifications
        for notification in notifications:
            self._execute_notification(notification)
        
        # Execute evidence collection
        if evidence.get("collection_type") != "minimal":
            self._execute_evidence_collection(evidence, state["detection"])
    
    def _execute_notification(self, notification: Dict[str, Any]):
        """Execute individual notification"""
        channel = notification.get("channel")
        message = notification.get("message")
        
        # Integration with legacy engine stubs
        if channel == "local":
            self.legacy_engine._beep_stub()
        elif channel == "operator":
            self.legacy_engine._notify_stub(
                notification.get("metadata", {}).get("detection_id"),
                "NOTIFY_OPERATOR",
                message,
                {}
            )
        elif channel == "uav":
            self.legacy_engine._dispatch_uav_stub(
                notification.get("metadata", {}).get("detection_id"),
                {}
            )
    
    def _execute_evidence_collection(self, evidence: Dict[str, Any], detection: Dict[str, Any]):
        """Execute evidence collection - DISABLED: Now handled by EvidenceAgent"""
        # Evidence collection is now handled by EvidenceAgent.process()
        # This stub is disabled to prevent duplicate evidence saving
        pass
    
    def _format_result(self, state: AgentState) -> Dict[str, Any]:
        """Format result to be compatible with original engine interface"""
        decision = state.get("decision", {})
        threat_assessment = state.get("threat_assessment", {})
        system_state = state.get("system_state", SystemState.NORMAL)
        emergency_response = state.get("emergency_response")
        
        return {
            "id": state["detection"].get("id"),
            "state": decision.get("response_level", system_state.value.upper()),
            "system_state": system_state.value,
            "state_changed": state.get("state_changed", False),
            "severity": threat_assessment.get("immediate_threat", {}).get("level", 0),
            "bayes_prob": threat_assessment.get("confidence_score", 0),
            "confidence": state["detection"].get("person_conf", 0),
            "duration_frames": state.get("memory_context", {}).get("detection_count", 0),
            "threat_score": decision.get("composite_score", 0),
            "action": "|".join(decision.get("actions", ["NO_ACTION"])),
            "reason": decision.get("rationale", "Agent-based decision"),
            "timestamp": state["timestamp"],
            "agent_messages": state.get("agent_messages", []),
            "notifications": state.get("notifications", []),
            "evidence_plan": state.get("evidence", {}),
            "emergency_response": emergency_response
        }

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Initialize agent-based engine
    agent_engine = AgentBasedDecisionEngine()
    
    # Test detection
    test_detection = {
        "id": 1,
        "bbox": [100, 100, 80, 200],
        "person_conf": 0.9,
        "gun_conf": 0.8,
        "knife_conf": 0.1,
        "fight_conf": 0.0,
        "meta": {"running": False},
        "timestamp": time.time()
    }
    
    print("=== Agent-Based Decision Engine Demo ===")
    result = agent_engine.process(test_detection)
    
    print(f"Detection ID: {result['id']}")
    print(f"Response Level: {result['state']}")
    print(f"Threat Score: {result['threat_score']:.2f}")
    print(f"Actions: {result['action']}")
    print(f"Reason: {result['reason']}")
    print(f"Agent Messages: {len(result['agent_messages'])}")
    print(f"Notifications: {len(result['notifications'])}")
