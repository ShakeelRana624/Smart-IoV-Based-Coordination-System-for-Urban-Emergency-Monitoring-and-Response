"""
Professional Memory Management System for Intelligent Weapon Detection System

Provides adaptive memory allocation, efficient buffer management, and automatic cleanup.
Optimizes memory usage for long-running security applications.
"""

import gc
import os
import psutil
import threading
import time
import weakref
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from .logging_system import get_logger


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    used_mb: float
    available_mb: float
    percentage: float
    buffer_usage_mb: float
    frame_count: int
    cleanup_count: int


class FrameBuffer:
    """
    Efficient frame buffer with memory management
    
    Features:
    - Adaptive sizing based on available memory
    - Automatic cleanup of old frames
    - Memory-efficient frame storage
    - Thread-safe operations
    """
    
    def __init__(self, camera_id: str, max_memory_mb: float = 256, max_age_seconds: int = 30):
        self.camera_id = camera_id
        self.max_memory_mb = max_memory_mb
        self.max_age_seconds = max_age_seconds
        self.logger = get_logger()
        
        # Frame storage with metadata
        self.frames = deque(maxlen=1000)  # Hard limit for safety
        self.current_memory_mb = 0.0
        self.frame_count = 0
        self.cleanup_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Memory tracking
        self.frame_sizes = []  # Track individual frame sizes
        self.max_frame_size_mb = 0.0
        
        # Adaptive parameters
        self.adaptive_max_frames = self._calculate_adaptive_limit()
        
        self.logger.info(f"FrameBuffer initialized for camera {camera_id}", extra={
            "component": "FrameBuffer",
            "camera_id": camera_id,
            "max_memory_mb": max_memory_mb,
            "adaptive_max_frames": self.adaptive_max_frames
        })
    
    def _calculate_adaptive_limit(self) -> int:
        """Calculate adaptive frame limit based on available memory"""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            # Reserve 25% of available memory for this buffer
            buffer_memory_mb = min(self.max_memory_mb, available_mb * 0.25)
            
            # Estimate frame size (640x480x3 bytes for RGB)
            estimated_frame_size_mb = (640 * 480 * 3) / (1024 * 1024)
            
            # Calculate maximum frames
            max_frames = int(buffer_memory_mb / estimated_frame_size_mb)
            
            # Ensure reasonable limits
            max_frames = max(10, min(max_frames, 500))  # Between 10 and 500 frames
            
            self.logger.debug(f"Calculated adaptive frame limit: {max_frames}", extra={
                "component": "FrameBuffer",
                "camera_id": self.camera_id,
                "available_mb": available_mb,
                "buffer_memory_mb": buffer_memory_mb,
                "estimated_frame_size_mb": estimated_frame_size_mb
            })
            
            return max_frames
            
        except Exception as e:
            self.logger.error("Failed to calculate adaptive limit", exception=e, extra={
                "component": "FrameBuffer",
                "camera_id": self.camera_id
            })
            return 100  # Safe default
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None) -> bool:
        """
        Add frame to buffer with memory management
        
        Args:
            frame: numpy array representing the frame
            timestamp: frame timestamp (default: current time)
            
        Returns:
            bool: True if frame was added successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            try:
                # Calculate frame size
                frame_size_mb = frame.nbytes / (1024 * 1024)
                
                # Check if we need to make space
                if self._needs_cleanup(frame_size_mb):
                    self._cleanup_old_frames()
                
                # Check memory limit
                if self.current_memory_mb + frame_size_mb > self.max_memory_mb:
                    # Force cleanup and retry
                    self._force_cleanup()
                    if self.current_memory_mb + frame_size_mb > self.max_memory_mb:
                        self.logger.warning("Frame buffer full, dropping frame", extra={
                            "component": "FrameBuffer",
                            "camera_id": self.camera_id,
                            "current_memory_mb": self.current_memory_mb,
                            "frame_size_mb": frame_size_mb,
                            "max_memory_mb": self.max_memory_mb
                        })
                        return False
                
                # Store frame efficiently
                frame_data = {
                    'frame': frame,  # Store reference, not copy
                    'timestamp': timestamp,
                    'size_mb': frame_size_mb
                }
                
                self.frames.append(frame_data)
                self.current_memory_mb += frame_size_mb
                self.frame_count += 1
                self.frame_sizes.append(frame_size_mb)
                
                # Track max frame size
                if frame_size_mb > self.max_frame_size_mb:
                    self.max_frame_size_mb = frame_size_mb
                
                # Adaptive adjustment
                if len(self.frames) > self.adaptive_max_frames:
                    self._adaptive_cleanup()
                
                return True
                
            except Exception as e:
                self.logger.error("Failed to add frame to buffer", exception=e, extra={
                    "component": "FrameBuffer",
                    "camera_id": self.camera_id
                })
                return False
    
    def _needs_cleanup(self, incoming_frame_size_mb: float) -> bool:
        """Check if cleanup is needed"""
        return (
            len(self.frames) >= self.adaptive_max_frames or
            self.current_memory_mb + incoming_frame_size_mb > self.max_memory_mb * 0.8 or
            self._has_old_frames()
        )
    
    def _has_old_frames(self) -> bool:
        """Check if buffer has frames older than max age"""
        if not self.frames:
            return False
        
        current_time = time.time()
        oldest_frame = self.frames[0]
        return (current_time - oldest_frame['timestamp']) > self.max_age_seconds
    
    def _cleanup_old_frames(self):
        """Remove old frames based on age and memory pressure"""
        if not self.frames:
            return
        
        current_time = time.time()
        frames_to_remove = []
        
        # Remove frames older than max age
        for i, frame_data in enumerate(self.frames):
            if current_time - frame_data['timestamp'] > self.max_age_seconds:
                frames_to_remove.append(i)
        
        # Remove frames from the front
        removed_count = 0
        for index in reversed(frames_to_remove):
            frame_data = self.frames[index]
            self.current_memory_mb -= frame_data['size_mb']
            del self.frames[index]
            removed_count += 1
        
        # Update frame sizes tracking
        if removed_count > 0:
            self.frame_sizes = self.frame_sizes[removed_count:]
            self.cleanup_count += 1
            
            self.logger.debug(f"Cleaned up {removed_count} old frames", extra={
                "component": "FrameBuffer",
                "camera_id": self.camera_id,
                "removed_count": removed_count,
                "current_memory_mb": self.current_memory_mb,
                "frame_count": len(self.frames)
            })
    
    def _force_cleanup(self):
        """Force aggressive cleanup to free memory"""
        if not self.frames:
            return
        
        # Remove oldest 50% of frames
        remove_count = len(self.frames) // 2
        removed_memory = 0.0
        
        for _ in range(remove_count):
            if self.frames:
                frame_data = self.frames.popleft()
                removed_memory += frame_data['size_mb']
        
        self.current_memory_mb -= removed_memory
        self.frame_sizes = self.frame_sizes[remove_count:]
        self.cleanup_count += 1
        
        # Force garbage collection
        gc.collect()
        
        self.logger.warning(f"Force cleanup: removed {remove_count} frames, freed {removed_memory:.2f}MB", extra={
            "component": "FrameBuffer",
            "camera_id": self.camera_id,
            "removed_count": remove_count,
            "freed_memory_mb": removed_memory,
            "current_memory_mb": self.current_memory_mb
        })
    
    def _adaptive_cleanup(self):
        """Adaptive cleanup based on memory pressure"""
        memory_pressure = self.current_memory_mb / self.max_memory_mb
        
        if memory_pressure > 0.9:
            # High pressure: remove 30%
            remove_count = len(self.frames) // 3
        elif memory_pressure > 0.7:
            # Medium pressure: remove 20%
            remove_count = len(self.frames) // 5
        else:
            # Low pressure: remove 10%
            remove_count = len(self.frames) // 10
        
        removed_memory = 0.0
        for _ in range(remove_count):
            if self.frames:
                frame_data = self.frames.popleft()
                removed_memory += frame_data['size_mb']
        
        self.current_memory_mb -= removed_memory
        self.frame_sizes = self.frame_sizes[remove_count:]
        self.cleanup_count += 1
        
        self.logger.debug(f"Adaptive cleanup: removed {remove_count} frames", extra={
            "component": "FrameBuffer",
            "camera_id": self.camera_id,
            "memory_pressure": memory_pressure,
            "removed_count": remove_count,
            "freed_memory_mb": removed_memory
        })
    
    def get_recent_frames(self, count: int = 10) -> List[np.ndarray]:
        """Get most recent frames"""
        with self.lock:
            recent_frames = []
            for frame_data in list(self.frames)[-count:]:
                recent_frames.append(frame_data['frame'])
            return recent_frames
    
    def get_frames_in_range(self, start_time: float, end_time: float) -> List[np.ndarray]:
        """Get frames within time range"""
        with self.lock:
            frames_in_range = []
            for frame_data in self.frames:
                if start_time <= frame_data['timestamp'] <= end_time:
                    frames_in_range.append(frame_data['frame'])
            return frames_in_range
    
    def get_stats(self) -> MemoryStats:
        """Get buffer statistics"""
        with self.lock:
            try:
                memory = psutil.virtual_memory()
                return MemoryStats(
                    total_mb=memory.total / (1024 * 1024),
                    used_mb=memory.used / (1024 * 1024),
                    available_mb=memory.available / (1024 * 1024),
                    percentage=memory.percent,
                    buffer_usage_mb=self.current_memory_mb,
                    frame_count=len(self.frames),
                    cleanup_count=self.cleanup_count
                )
            except Exception as e:
                self.logger.error("Failed to get memory stats", exception=e, extra={
                    "component": "FrameBuffer",
                    "camera_id": self.camera_id
                })
                return MemoryStats(0, 0, 0, 0, self.current_memory_mb, len(self.frames), self.cleanup_count)
    
    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            self.frames.clear()
            self.current_memory_mb = 0.0
            self.frame_sizes.clear()
            self.frame_count = 0
            
            self.logger.info(f"FrameBuffer cleared for camera {self.camera_id}", extra={
                "component": "FrameBuffer",
                "camera_id": self.camera_id
            })


class MemoryManager:
    """
    Central memory management system
    
    Features:
    - Multiple camera buffer management
    - System-wide memory monitoring
    - Automatic cleanup and optimization
    - Memory pressure handling
    """
    
    def __init__(self, max_system_memory_mb: float = 2048):
        self.max_system_memory_mb = max_system_memory_mb
        self.logger = get_logger()
        
        # Camera buffers
        self.buffers: Dict[str, FrameBuffer] = {}
        self.buffer_lock = threading.RLock()
        
        # Memory monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 30  # seconds
        
        # Statistics
        self.total_cleanup_count = 0
        self.memory_pressure_events = 0
        
        # Memory thresholds
        self.warning_threshold = 0.7  # 70%
        self.critical_threshold = 0.85  # 85%
        self.emergency_threshold = 0.95  # 95%
        
        self.logger.info("MemoryManager initialized", extra={
            "component": "MemoryManager",
            "max_system_memory_mb": max_system_memory_mb
        })
    
    def create_buffer(self, camera_id: str, max_memory_mb: float = 256) -> FrameBuffer:
        """Create frame buffer for camera"""
        with self.buffer_lock:
            if camera_id in self.buffers:
                self.logger.warning(f"Buffer already exists for camera {camera_id}", extra={
                    "component": "MemoryManager",
                    "camera_id": camera_id
                })
                return self.buffers[camera_id]
            
            # Calculate per-camera memory limit
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / (1024 * 1024)
            
            # Limit per-camera buffer based on available memory
            per_camera_limit = min(max_memory_mb, available_mb * 0.1)
            
            buffer = FrameBuffer(camera_id, per_camera_limit)
            self.buffers[camera_id] = buffer
            
            self.logger.info(f"Created buffer for camera {camera_id}", extra={
                "component": "MemoryManager",
                "camera_id": camera_id,
                "allocated_memory_mb": per_camera_limit
            })
            
            return buffer
    
    def get_buffer(self, camera_id: str) -> Optional[FrameBuffer]:
        """Get frame buffer for camera"""
        with self.buffer_lock:
            return self.buffers.get(camera_id)
    
    def remove_buffer(self, camera_id: str):
        """Remove frame buffer for camera"""
        with self.buffer_lock:
            if camera_id in self.buffers:
                buffer = self.buffers[camera_id]
                buffer.clear()
                del self.buffers[camera_id]
                
                self.logger.info(f"Removed buffer for camera {camera_id}", extra={
                    "component": "MemoryManager",
                    "camera_id": camera_id
                })
    
    def add_frame(self, camera_id: str, frame: np.ndarray, timestamp: float = None) -> bool:
        """Add frame to camera buffer"""
        buffer = self.get_buffer(camera_id)
        if buffer is None:
            buffer = self.create_buffer(camera_id)
        
        return buffer.add_frame(frame, timestamp)
    
    def get_system_memory_stats(self) -> MemoryStats:
        """Get system-wide memory statistics"""
        try:
            memory = psutil.virtual_memory()
            
            # Calculate total buffer usage
            total_buffer_usage = 0.0
            total_frames = 0
            total_cleanups = 0
            
            with self.buffer_lock:
                for buffer in self.buffers.values():
                    stats = buffer.get_stats()
                    total_buffer_usage += stats.buffer_usage_mb
                    total_frames += stats.frame_count
                    total_cleanups += stats.cleanup_count
            
            return MemoryStats(
                total_mb=memory.total / (1024 * 1024),
                used_mb=memory.used / (1024 * 1024),
                available_mb=memory.available / (1024 * 1024),
                percentage=memory.percent,
                buffer_usage_mb=total_buffer_usage,
                frame_count=total_frames,
                cleanup_count=total_cleanups
            )
            
        except Exception as e:
            self.logger.error("Failed to get system memory stats", exception=e, extra={
                "component": "MemoryManager"
            })
            return MemoryStats(0, 0, 0, 0, 0, 0, 0)
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Memory monitoring started", extra={
            "component": "MemoryManager"
        })
    
    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Memory monitoring stopped", extra={
            "component": "MemoryManager"
        })
    
    def _monitor_memory(self):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self.get_system_memory_stats()
                
                # Check memory pressure
                if stats.percentage >= self.emergency_threshold:
                    self._handle_emergency_memory()
                elif stats.percentage >= self.critical_threshold:
                    self._handle_critical_memory()
                elif stats.percentage >= self.warning_threshold:
                    self._handle_warning_memory()
                
                # Log memory stats periodically
                self.logger.debug(f"Memory usage: {stats.percentage:.1f}%", extra={
                    "component": "MemoryManager",
                    "memory_percentage": stats.percentage,
                    "buffer_usage_mb": stats.buffer_usage_mb,
                    "total_frames": stats.frame_count
                })
                
            except Exception as e:
                self.logger.error("Memory monitoring error", exception=e, extra={
                    "component": "MemoryManager"
                })
            
            time.sleep(self.monitor_interval)
    
    def _handle_warning_memory(self):
        """Handle warning level memory pressure"""
        self.memory_pressure_events += 1
        
        # Trigger cleanup in all buffers
        with self.buffer_lock:
            for buffer in self.buffers.values():
                buffer._cleanup_old_frames()
        
        self.logger.warning("Memory pressure warning - cleanup triggered", extra={
            "component": "MemoryManager",
            "memory_pressure_events": self.memory_pressure_events
        })
    
    def _handle_critical_memory(self):
        """Handle critical level memory pressure"""
        self.memory_pressure_events += 1
        
        # Force cleanup in all buffers
        with self.buffer_lock:
            for buffer in self.buffers.values():
                buffer._force_cleanup()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.critical("Critical memory pressure - force cleanup", extra={
            "component": "MemoryManager",
            "memory_pressure_events": self.memory_pressure_events
        })
    
    def _handle_emergency_memory(self):
        """Handle emergency level memory pressure"""
        self.memory_pressure_events += 1
        
        # Clear all buffers
        with self.buffer_lock:
            for buffer in self.buffers.values():
                buffer.clear()
        
        # Aggressive garbage collection
        for _ in range(3):
            gc.collect()
        
        self.logger.critical("Emergency memory pressure - all buffers cleared", extra={
            "component": "MemoryManager",
            "memory_pressure_events": self.memory_pressure_events
        })
    
    def get_buffer_stats(self) -> Dict[str, MemoryStats]:
        """Get statistics for all buffers"""
        with self.buffer_lock:
            stats = {}
            for camera_id, buffer in self.buffers.items():
                stats[camera_id] = buffer.get_stats()
            return stats
    
    def cleanup_all(self):
        """Cleanup all buffers"""
        with self.buffer_lock:
            for buffer in self.buffers.values():
                buffer.clear()
        
        self.logger.info("All buffers cleaned up", extra={
            "component": "MemoryManager"
        })
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_monitoring()
        self.cleanup_all()


# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        _memory_manager.start_monitoring()
    return _memory_manager

def setup_memory_manager(max_system_memory_mb: float = 2048) -> MemoryManager:
    """Setup memory manager with configuration"""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.stop_monitoring()
    
    _memory_manager = MemoryManager(max_system_memory_mb)
    _memory_manager.start_monitoring()
    return _memory_manager
