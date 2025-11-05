# Configuration settings for the BDA_9_Recommendation project

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model parameters
MODEL_TYPE = 'collaborative_filtering'  # Options: 'collaborative_filtering', 'content_based'
NUM_RECOMMENDATIONS = 10

# Logging settings
LOGGING_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# Other configurations
RANDOM_SEED = 42