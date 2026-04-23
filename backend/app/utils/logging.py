import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.config import settings

LOG_DIR = Path(settings.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("decision_assistant")
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Rotating file handler
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(file_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)