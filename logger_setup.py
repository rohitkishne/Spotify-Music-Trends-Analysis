# Import Logger
import logging

# Configure Logging
logging.basicConfig(
    filename="spotify_analysis.log",
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s",
    filemode='w'
)