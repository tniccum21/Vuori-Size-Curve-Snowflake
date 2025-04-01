"""
Logging configuration for size curve analysis.
"""
import os
import logging
from datetime import datetime
import threading

# Track if logging has been initialized
_logging_initialized = False
_logging_lock = threading.Lock()


def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: "logs")
    
    Returns:
        logging.Logger: Configured logger
    """
    global _logging_initialized
    
    # Use lock to prevent race conditions when multiple threads try to initialize logging
    with _logging_lock:
        if _logging_initialized:
            return logging.getLogger()
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate a timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"size_curve_analysis_{timestamp}.log")
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        _logging_initialized = True
        
        return logger


def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module (typically __name__)
    
    Returns:
        logging.Logger: Logger instance
    """
    # Initialize root logger if not done already
    if not _logging_initialized:
        setup_logging()
    
    # Get a logger for the specified name
    logger = logging.getLogger(name)
    
    return logger