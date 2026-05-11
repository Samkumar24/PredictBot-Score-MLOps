import yaml
from pathlib import Path
from ensure import ensure_annotations
# New/Correct way
from box import ConfigBox
from src.predictor_bot_score.logger import logger

@ensure_annotations
def yaml_load(pathlib : Path):
    try:    
        with open(pathlib) as yaml_file:
            yaml_content = yaml.safe_load(yaml_file) 
            logger.info(f"yaml file: {pathlib} loaded successfully")
        return ConfigBox(yaml_content)
    except Exception as e:
        raise ValueError(f"Error occurred while reading yaml file: {pathlib}. Error: {e}")
        


