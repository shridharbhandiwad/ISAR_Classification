"""Logging utilities for ISAR Image Analysis."""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "isar_analysis",
    log_dir: str = "logs",
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up and configure the logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger(name: str = "isar_analysis") -> logging.Logger:
    """
    Get the logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logger(name)
    
    return _logger


class LoggerContext:
    """Context manager for logging training progress."""
    
    def __init__(self, logger: logging.Logger, phase: str):
        self.logger = logger
        self.phase = phase
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.phase}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.phase} in {duration}")
        else:
            self.logger.error(f"Failed {self.phase} after {duration}: {exc_val}")
        return False
