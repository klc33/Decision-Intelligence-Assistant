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

# File handler (UTF-8, no console issues)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(file_format)

# Console handler – use plain ASCII to avoid encoding problems
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)