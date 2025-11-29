import logging
import os

def setup_logging(log_file="debug.log"):
    """
    Configures the root logger to write to a file.
    
    Args:
        log_file (str): The path to the log file.
    """
    # Create the log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode='w', # Overwrite log on each run
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a logger for this module to verify setup
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully.")
