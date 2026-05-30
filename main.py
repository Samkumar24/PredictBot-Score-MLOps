from src.predictor_bot_score.components._01_data_injestion  import Data_injestion
from src.predictor_bot_score.entity import Data_injestion_config
from src.predictor_bot_score.config.configuration import yaml_configruation


con = yaml_configruation()
con_data_injestion = con.get_data_ingestion_config()
con_data_injestion = Data_injestion(con_data_injestion)
con_data_injestion.s3_downlaod_data()
