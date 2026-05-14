from src.predictor_bot_score.utils.common import yaml_load , create_directories
from pathlib import Path

pathlib = Path("config\config.yaml")
print(pathlib)
x = yaml_load(pathlib)
print(x.data_ingestion)
# Or even deeper

#create_directories([x.data_ingestion.raw_data,x.data_ingestion.ingested_data])