import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler


def setup_logger(name: str, log_dir: Path, level: str = "INFO"):
    """Setup comprehensive logging with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    # File handler
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with Rich
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
