import lightgbm as lgb
#import xgboost as xgb
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import yaml
from pathlib import Path
from model_factory import get_model , MODEL_REGISTRY , get_fit_kwargs ,EARLY_STOPPING_REGISTRY
import pandas as pd 


CONFIG_PATH = Path('config/config.yaml')
print(CONFIG_PATH)
val_set_path = Path(r'artifacts\feature_engineering\featured\Validation_data\val_2026_06_15_11_48.csv')
val_data = pd.read_csv(val_set_path)

X_val = val_data.drop(columns=['bot_score'])
y_val = val_data['bot_score']
my_eval_set = [(X_val, y_val)]

with open(CONFIG_PATH ,'r') as f:
    config = yaml.safe_load(f)

model_config = config['model_training']['models']


for model_name , config in model_config.items():
    model_type = config['type']
    model_params = config['params']

    if model_type not in MODEL_REGISTRY:
        print(f"Skipping {model_name}: {model_type} not in registry.")
        continue

    model = get_model(model_type , model_params)
    #print(model)

    fit_kwargs = get_fit_kwargs(model_type,my_eval_set)
    print(fit_kwargs)


