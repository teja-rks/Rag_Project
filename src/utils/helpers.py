import logging
import json
import sys
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Create a structured dictionary for the log
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }
        
        # Include any extra attributes passed in dynamically
        if hasattr(record, 'extra_info'):
            log_record['extra_info'] = record.extra_info
            
        return json.dumps(log_record)

def get_logger(name: str) -> logging.Logger:
    """
    Initializes and returns a structured JSON logger.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if the logger is requested multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File Handler: Routes JSON logs to the app.log file
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(JsonFormatter())
        
        # Stream Handler: Prints logs to the console for local debugging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger