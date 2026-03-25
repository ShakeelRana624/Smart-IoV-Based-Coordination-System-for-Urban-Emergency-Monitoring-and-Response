"""
Professional Logging System for Intelligent Weapon Detection System

Provides structured, production-ready logging with multiple handlers,
log rotation, and performance monitoring.
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import threading
from enum import Enum


class LogLevel(Enum):
    """Log levels with severity mapping"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class SecurityLogger:
    """
    Professional logging system for security monitoring
    
    Features:
    - Multiple output handlers (console, file, remote)
    - Log rotation with size limits
    - Structured JSON logging for machine processing
    - Performance metrics tracking
    - Thread-safe operations
    - Context-aware logging
    """
    
    def __init__(self, name: str = "security_system", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.config.get("level", logging.INFO))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Performance tracking
        self.metrics = {
            "total_logs": 0,
            "error_count": 0,
            "warning_count": 0,
            "start_time": time.time()
        }
        self.metrics_lock = threading.Lock()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler()
        self._setup_json_handler()
        self._setup_error_handler()
        
        # Log initialization
        self.info("Security logging system initialized", extra={
            "component": "LoggingSystem",
            "config": self.config
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        base_dir = Path(__file__).parent.parent
        return {
            "level": logging.INFO,
            "console_level": logging.WARNING,
            "file_level": logging.INFO,
            "json_level": logging.DEBUG,
            "log_dir": base_dir / "logs",
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "enable_performance_tracking": True
        }
    
    def _setup_console_handler(self):
        """Setup console handler for real-time monitoring"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.get("console_level", logging.WARNING))
        
        # Simple formatter for console
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt=self.config.get("date_format", "%H:%M:%S")
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler for persistent logs"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "security.log",
            maxBytes=self.config.get("max_file_size", 10*1024*1024),
            backupCount=self.config.get("backup_count", 5),
            encoding='utf-8'
        )
        file_handler.setLevel(self.config.get("file_level", logging.INFO))
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            self.config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            datefmt=self.config.get("date_format", "%Y-%m-%d %H:%M:%S")
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self):
        """Setup JSON handler for machine-readable logs"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "security.json",
            maxBytes=self.config.get("max_file_size", 10*1024*1024),
            backupCount=self.config.get("backup_count", 5),
            encoding='utf-8'
        )
        json_handler.setLevel(self.config.get("json_level", logging.DEBUG))
        
        # JSON formatter for structured logging
        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(json_handler)
    
    def _setup_error_handler(self):
        """Setup separate handler for errors and critical events"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=self.config.get("max_file_size", 10*1024*1024),
            backupCount=self.config.get("backup_count", 5),
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # Detailed error formatter with traceback
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n"
            "Exception: %(exc_info)s\n",
            datefmt=self.config.get("date_format", "%Y-%m-%d %H:%M:%S")
        )
        error_handler.setFormatter(error_formatter)
        
        self.logger.addHandler(error_handler)
    
    def _update_metrics(self, level: int):
        """Update performance metrics"""
        if not self.config.get("enable_performance_tracking", True):
            return
        
        with self.metrics_lock:
            self.metrics["total_logs"] += 1
            if level >= logging.ERROR:
                self.metrics["error_count"] += 1
            elif level >= logging.WARNING:
                self.metrics["warning_count"] += 1
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception"""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with context"""
        # Update metrics
        self._update_metrics(level)
        
        # Extract structured data
        extra = kwargs.get("extra", {})
        component = kwargs.get("component", "Unknown")
        camera_id = kwargs.get("camera_id", None)
        detection_id = kwargs.get("detection_id", None)
        
        # Add context to log record
        log_extra = {
            "component": component,
            "camera_id": camera_id,
            "detection_id": detection_id,
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid()
        }
        
        # Add any additional context
        log_extra.update(extra)
        
        # Log the message
        self.logger.log(level, message, extra=log_extra)
    
    def detection_event(self, detection_type: str, confidence: float, camera_id: str, **kwargs):
        """Log detection event with structured data"""
        self.info(f"Detection event: {detection_type}", extra={
            "event_type": "detection",
            "detection_type": detection_type,
            "confidence": confidence,
            "camera_id": camera_id,
            **kwargs
        })
    
    def system_event(self, event_type: str, **kwargs):
        """Log system event with structured data"""
        self.info(f"System event: {event_type}", extra={
            "event_type": "system",
            "system_event": event_type,
            **kwargs
        })
    
    def performance_event(self, operation: str, duration: float, **kwargs):
        """Log performance event"""
        self.debug(f"Performance: {operation} took {duration:.3f}s", extra={
            "event_type": "performance",
            "operation": operation,
            "duration": duration,
            **kwargs
        })
    
    def security_event(self, threat_level: str, **kwargs):
        """Log security event"""
        level = getattr(LogLevel, threat_level.upper(), LogLevel.WARNING).value
        self._log(level, f"Security event: {threat_level}", extra={
            "event_type": "security",
            "threat_level": threat_level,
            **kwargs
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.metrics_lock:
            uptime = time.time() - self.metrics["start_time"]
            return {
                **self.metrics,
                "uptime_seconds": uptime,
                "logs_per_second": self.metrics["total_logs"] / uptime if uptime > 0 else 0
            }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self.metrics_lock:
            self.metrics = {
                "total_logs": 0,
                "error_count": 0,
                "warning_count": 0,
                "start_time": time.time()
            }


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process
        }
        
        # Add structured extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


# Global logger instance
_security_logger = None

def get_logger(name: str = "security_system", config: Optional[Dict[str, Any]] = None) -> SecurityLogger:
    """Get or create logger instance"""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityLogger(name, config)
    return _security_logger

def setup_logging(config: Optional[Dict[str, Any]] = None) -> SecurityLogger:
    """Setup logging system with configuration"""
    return get_logger("security_system", config)


# Context managers for performance monitoring
class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation: str, logger: Optional[SecurityLogger] = None, **kwargs):
        self.operation = operation
        self.logger = logger or get_logger()
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.performance_event(self.operation, duration, **self.kwargs)


# Convenience functions
def log_detection(detection_type: str, confidence: float, camera_id: str, **kwargs):
    """Convenience function for detection logging"""
    get_logger().detection_event(detection_type, confidence, camera_id, **kwargs)

def log_system_event(event_type: str, **kwargs):
    """Convenience function for system event logging"""
    get_logger().system_event(event_type, **kwargs)

def log_security_event(threat_level: str, **kwargs):
    """Convenience function for security event logging"""
    get_logger().security_event(threat_level, **kwargs)
