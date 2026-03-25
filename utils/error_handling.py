"""
Professional Error Handling System for Intelligent Weapon Detection System

Provides specific exception types, graceful error recovery, and comprehensive error reporting.
Replaces generic exception handling with precise, actionable error management.
"""

import errno
import os
import sys
import traceback
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Type
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from .logging_system import get_logger


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    SYSTEM = "system"
    CAMERA = "camera"
    MODEL = "model"
    STORAGE = "storage"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    DETECTION = "detection"


# Base exception class
class SecuritySystemError(Exception):
    """Base exception for security system"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, **context):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.timestamp = datetime.now()
        self.thread_id = threading.current_thread().ident
        self.process_id = os.getpid()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "thread_id": self.thread_id,
            "process_id": self.process_id,
            "traceback": traceback.format_exc()
        }


# Specific exception types
class ConfigurationError(SecuritySystemError):
    """Configuration related errors"""
    
    def __init__(self, message: str, **context):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, **context)


class CameraError(SecuritySystemError):
    """Camera related errors"""
    
    def __init__(self, message: str, camera_id: str = None, **context):
        context["camera_id"] = camera_id
        super().__init__(message, ErrorCategory.CAMERA, ErrorSeverity.HIGH, **context)


class ModelError(SecuritySystemError):
    """AI model related errors"""
    
    def __init__(self, message: str, model_name: str = None, **context):
        context["model_name"] = model_name
        super().__init__(message, ErrorCategory.MODEL, ErrorSeverity.CRITICAL, **context)


class StorageError(SecuritySystemError):
    """Storage related errors"""
    
    def __init__(self, message: str, storage_path: str = None, **context):
        context["storage_path"] = storage_path
        super().__init__(message, ErrorCategory.STORAGE, ErrorSeverity.MEDIUM, **context)


class NetworkError(SecuritySystemError):
    """Network related errors"""
    
    def __init__(self, message: str, endpoint: str = None, **context):
        context["endpoint"] = endpoint
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, **context)


class MemoryError(SecuritySystemError):
    """Memory related errors"""
    
    def __init__(self, message: str, memory_usage_mb: float = None, **context):
        context["memory_usage_mb"] = memory_usage_mb
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.HIGH, **context)


class DetectionError(SecuritySystemError):
    """Detection related errors"""
    
    def __init__(self, message: str, detection_type: str = None, **context):
        context["detection_type"] = detection_type
        super().__init__(message, ErrorCategory.DETECTION, ErrorSeverity.LOW, **context)


@dataclass
class ErrorReport:
    """Comprehensive error report"""
    error_type: str
    message: str
    category: str
    severity: str
    context: Dict[str, Any]
    timestamp: datetime
    thread_id: int
    process_id: int
    traceback: str
    recovery_action: Optional[str] = None
    resolution_status: str = "pending"


class ErrorHandler:
    """
    Professional error handling and recovery system
    
    Features:
    - Specific exception handling
    - Automatic recovery attempts
    - Error reporting and tracking
    - Graceful degradation
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.error_reports: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_error_reports = 1000
        self.error_report_lock = threading.Lock()
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        self.logger.info("ErrorHandler initialized", extra={
            "component": "ErrorHandler"
        })
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.recovery_strategies.update({
            CameraError: self._handle_camera_error,
            ModelError: self._handle_model_error,
            StorageError: self._handle_storage_error,
            NetworkError: self._handle_network_error,
            MemoryError: self._handle_memory_error,
            ConfigurationError: self._handle_configuration_error,
        })
    
    def handle_exception(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorReport:
        """
        Handle exception with appropriate recovery strategy
        
        Args:
            exception: The exception to handle
            context: Additional context information
            
        Returns:
            ErrorReport with handling details
        """
        context = context or {}
        
        # Convert to SecuritySystemError if needed
        if not isinstance(exception, SecuritySystemError):
            security_error = SecuritySystemError(
                str(exception),
                ErrorCategory.SYSTEM,
                ErrorSeverity.MEDIUM,
                **context
            )
        else:
            security_error = exception
        
        # Create error report
        error_report = ErrorReport(
            error_type=security_error.__class__.__name__,
            message=security_error.message,
            category=security_error.category.value,
            severity=security_error.severity.value,
            context=security_error.context,
            timestamp=security_error.timestamp,
            thread_id=security_error.thread_id,
            process_id=security_error.process_id,
            traceback=traceback.format_exc()
        )
        
        # Log error appropriately
        self._log_error(security_error)
        
        # Attempt recovery
        recovery_action = self._attempt_recovery(security_error)
        error_report.recovery_action = recovery_action
        
        # Update error tracking
        self._update_error_tracking(error_report)
        
        # Store error report
        self._store_error_report(error_report)
        
        return error_report
    
    def _log_error(self, error: SecuritySystemError):
        """Log error with appropriate level"""
        log_method = {
            ErrorSeverity.LOW: self.logger.warning,
            ErrorSeverity.MEDIUM: self.logger.error,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }.get(error.severity, self.logger.error)
        
        log_method(
            f"[{error.category.value.upper()}] {error.message}",
            extra={
                "component": "ErrorHandler",
                "error_category": error.category.value,
                "error_severity": error.severity.value,
                **error.context
            }
        )
    
    def _attempt_recovery(self, error: SecuritySystemError) -> Optional[str]:
        """Attempt recovery based on error type"""
        recovery_strategy = self.recovery_strategies.get(type(error))
        if recovery_strategy:
            try:
                return recovery_strategy(error)
            except Exception as e:
                self.logger.error("Recovery strategy failed", exception=e, extra={
                    "component": "ErrorHandler",
                    "original_error": error.message
                })
                return f"Recovery failed: {str(e)}"
        
        return None
    
    def _handle_camera_error(self, error: CameraError) -> str:
        """Handle camera error with recovery"""
        camera_id = error.context.get("camera_id", "unknown")
        
        # Recovery strategies for camera errors
        if "connection" in error.message.lower():
            return f"Attempting camera reconnection for {camera_id}"
        elif "permission" in error.message.lower():
            return f"Checking camera permissions for {camera_id}"
        elif "not found" in error.message.lower():
            return f"Searching for available cameras to replace {camera_id}"
        else:
            return f"Restarting camera {camera_id}"
    
    def _handle_model_error(self, error: ModelError) -> str:
        """Handle model error with recovery"""
        model_name = error.context.get("model_name", "unknown")
        
        if "not found" in error.message.lower():
            return f"Searching for alternative model for {model_name}"
        elif "corrupted" in error.message.lower():
            return f"Attempting model reload for {model_name}"
        elif "memory" in error.message.lower():
            return "Optimizing model memory usage"
        else:
            return f"Reinitializing model {model_name}"
    
    def _handle_storage_error(self, error: StorageError) -> str:
        """Handle storage error with recovery"""
        storage_path = error.context.get("storage_path", "unknown")
        
        if "permission" in error.message.lower():
            return f"Checking storage permissions for {storage_path}"
        elif "space" in error.message.lower():
            return "Attempting storage cleanup"
        elif "not found" in error.message.lower():
            return f"Creating storage directory {storage_path}"
        else:
            return f"Switching to alternative storage for {storage_path}"
    
    def _handle_network_error(self, error: NetworkError) -> str:
        """Handle network error with recovery"""
        endpoint = error.context.get("endpoint", "unknown")
        
        if "timeout" in error.message.lower():
            return f"Increasing timeout for {endpoint}"
        elif "connection" in error.message.lower():
            return f"Attempting reconnection to {endpoint}"
        elif "dns" in error.message.lower():
            return "Using alternative DNS configuration"
        else:
            return f"Switching to offline mode for {endpoint}"
    
    def _handle_memory_error(self, error: MemoryError) -> str:
        """Handle memory error with recovery"""
        memory_usage = error.context.get("memory_usage_mb", 0)
        
        if "allocation" in error.message.lower():
            return "Optimizing memory allocation"
        elif "leak" in error.message.lower():
            return "Triggering garbage collection"
        else:
            return f"Clearing memory buffers (current: {memory_usage}MB)"
    
    def _handle_configuration_error(self, error: ConfigurationError) -> str:
        """Handle configuration error with recovery"""
        if "not found" in error.message.lower():
            return "Loading default configuration"
        elif "invalid" in error.message.lower():
            return "Resetting configuration to defaults"
        else:
            return "Reloading configuration from file"
    
    def _update_error_tracking(self, error_report: ErrorReport):
        """Update error tracking statistics"""
        with self.error_report_lock:
            error_key = f"{error_report.error_type}:{error_report.category}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _store_error_report(self, error_report: ErrorReport):
        """Store error report with size limit"""
        with self.error_report_lock:
            self.error_reports.append(error_report)
            
            # Maintain size limit
            if len(self.error_reports) > self.max_error_reports:
                self.error_reports = self.error_reports[-self.max_error_reports:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.error_report_lock:
            total_errors = len(self.error_reports)
            error_by_category = {}
            error_by_severity = {}
            recent_errors = []
            
            for report in self.error_reports:
                # Category statistics
                category = report.category
                error_by_category[category] = error_by_category.get(category, 0) + 1
                
                # Severity statistics
                severity = report.severity
                error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
                
                # Recent errors (last hour)
                if (datetime.now() - report.timestamp).total_seconds() < 3600:
                    recent_errors.append(report)
            
            return {
                "total_errors": total_errors,
                "error_counts": dict(self.error_counts),
                "error_by_category": error_by_category,
                "error_by_severity": error_by_severity,
                "recent_errors_count": len(recent_errors),
                "most_common_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
    
    def get_recent_errors(self, count: int = 10, category: str = None) -> List[ErrorReport]:
        """Get recent errors with optional category filter"""
        with self.error_report_lock:
            errors = self.error_reports.copy()
            
            if category:
                errors = [e for e in errors if e.category == category]
            
            return sorted(errors, key=lambda x: x.timestamp, reverse=True)[:count]


# Decorator for automatic error handling
def handle_errors(
    default_return=None,
    swallow_errors: bool = False,
    log_errors: bool = True,
    error_category: ErrorCategory = ErrorCategory.SYSTEM
):
    """
    Decorator for automatic error handling
    
    Args:
        default_return: Default return value on error
        swallow_errors: Whether to swallow exceptions
        log_errors: Whether to log errors
        error_category: Default error category
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler
                error_handler = get_error_handler()
                
                # Create context
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                # Handle exception
                error_report = error_handler.handle_exception(e, context)
                
                if log_errors:
                    error_handler.logger.error(f"Error in {func.__name__}", extra={
                        "component": "ErrorDecorator",
                        "function": func.__name__,
                        "error_report": error_report.__dict__
                    })
                
                if swallow_errors:
                    return default_return
                else:
                    raise e
        return wrapper
    return decorator


# Specific error handling functions
def handle_file_operation(operation: str, file_path: str, **kwargs):
    """Handle file operations with specific error handling"""
    try:
        return operation(file_path, **kwargs)
    except FileNotFoundError:
        raise StorageError(f"File not found: {file_path}", storage_path=file_path, **kwargs)
    except PermissionError:
        raise StorageError(f"Permission denied: {file_path}", storage_path=file_path, **kwargs)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise StorageError(f"No disk space: {file_path}", storage_path=file_path, **kwargs)
        elif e.errno == errno.EACCES:
            raise StorageError(f"Access denied: {file_path}", storage_path=file_path, **kwargs)
        else:
            raise StorageError(f"OS error ({e.errno}): {file_path}", storage_path=file_path, **kwargs)


def handle_camera_operation(operation: str, camera_id: str, **kwargs):
    """Handle camera operations with specific error handling"""
    try:
        return operation(camera_id, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "no device" in error_msg:
            raise CameraError(f"Camera not found: {camera_id}", camera_id=camera_id, **kwargs)
        elif "permission" in error_msg or "access denied" in error_msg:
            raise CameraError(f"Camera access denied: {camera_id}", camera_id=camera_id, **kwargs)
        elif "busy" in error_msg or "in use" in error_msg:
            raise CameraError(f"Camera busy: {camera_id}", camera_id=camera_id, **kwargs)
        else:
            raise CameraError(f"Camera error: {camera_id} - {str(e)}", camera_id=camera_id, **kwargs)


def handle_model_operation(operation: str, model_name: str, **kwargs):
    """Handle model operations with specific error handling"""
    try:
        return operation(model_name, **kwargs)
    except FileNotFoundError:
        raise ModelError(f"Model not found: {model_name}", model_name=model_name, **kwargs)
    except MemoryError:
        raise ModelError(f"Model memory error: {model_name}", model_name=model_name, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "corrupted" in error_msg or "invalid" in error_msg:
            raise ModelError(f"Model corrupted: {model_name}", model_name=model_name, **kwargs)
        else:
            raise ModelError(f"Model error: {model_name} - {str(e)}", model_name=model_name, **kwargs)


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def setup_error_handler() -> ErrorHandler:
    """Setup error handler"""
    global _error_handler
    _error_handler = ErrorHandler()
    return _error_handler


# Convenience functions
def log_error(message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
              severity: ErrorSeverity = ErrorSeverity.MEDIUM, **context):
    """Log error with structured data"""
    error_handler = get_error_handler()
    error = SecuritySystemError(message, category, severity, **context)
    error_handler.handle_exception(error)

def handle_safe_operation(operation: Callable, *args, **kwargs):
    """Safely execute operation with error handling"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        error_handler = get_error_handler()
        error_handler.handle_exception(e, {
            "operation": operation.__name__,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        })
        return None
