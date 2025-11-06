import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler


def setup_logger(name: str, log_dir: Path = Path("./logs"), level: str = "INFO"):
    """Setup logger with file and console handlers - FIXED for Unicode"""
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - FIXED: Use UTF-8 encoding
    log_file = log_dir / f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(
        log_file, 
        mode='a', 
        encoding='utf-8'  # FIXED: Explicitly use UTF-8
    )
    file_handler.setLevel(logging.DEBUG)
    
    # File formatter - FIXED: Use ASCII-safe symbols
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (Rich) - Already handles Unicode well
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
