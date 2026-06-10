from src.predictor_bot_score.components._01_data_injestion  import Data_injestion
from src.predictor_bot_score.components._02_data_validation  import Data_validation
from src.predictor_bot_score.entity import (Data_injestion_config, DataValidationConfig)
from src.predictor_bot_score.config.configuration import (yaml_configruation)
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.components._01_data_injestion  import Data_injestion
from src.predictor_bot_score.entity import Data_injestion_config
from src.predictor_bot_score.config.configuration import yaml_configruation




# --- Execution Script ---

# 1. Data Ingestion Stage
logger.info("--- Stage 1: Data Ingestion ---")
ingestion_config = yaml_configruation().get_data_ingestion_config()
ingestion_manager = Data_injestion(ingestion_config)
ingestion_manager.s3_downlaod_data()

# 2. Data Validation Stage
logger.info("--- Stage 2: Data Validation ---")
validation_config = yaml_configruation().get_data_validation_config()
validation_manager = Data_validation(validation_config)
validation_manager.run() # This single call handles all 5 stages!

