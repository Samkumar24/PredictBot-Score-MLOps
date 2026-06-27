from src.predictor_bot_score.components._01_data_injestion  import Data_injestion
from src.predictor_bot_score.components._02_data_validation  import Data_validation
from src.predictor_bot_score.components._03_data_transformation  import Data_transformation
from src.predictor_bot_score.components._04_feature_enginerring import Feature_engineering
from src.predictor_bot_score.components._05_model_building import Model_Building
from src.predictor_bot_score.components._06_model_evalulation import Model_evalulation
from src.predictor_bot_score.config.configuration import (yaml_configruation)
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.entity import Data_injestion_config
from src.predictor_bot_score.config.configuration import yaml_configruation




logger.info("--- Stage 6: Model Evaluation---")
model_eval_config = yaml_configruation().get_model_evaluation_config()
model_eval_config = Model_evalulation(model_eval_config)
model_eval_config.run()