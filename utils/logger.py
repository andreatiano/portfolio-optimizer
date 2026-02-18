"""Logger configurato per il progetto."""
import logging
import os
from datetime import datetime


def setup_logger(name="portfolio_optimizer", level=logging.INFO):
    os.makedirs("output", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # Solo warning/error su console
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        # File handler
        log_file = f"output/run_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
