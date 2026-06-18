# src/predictor_bot_score/utils/model_factory.py

import lightgbm as lgb
import xgboost as xgb
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import yaml
from pathlib import Path


# ── MODEL REGISTRY ────────────────────────────────────
#
# KEY   = string written in config.yaml under "type"
# VALUE = actual Python class for that model
#
# model_training.py reads "type: lightgbm" from YAML
# calls get_model("lightgbm", params)
# this registry returns LGBMRegressor(**params)
#
# TO ADD NEW MODEL:
#   1. import library at top of this file
#   2. add one line to MODEL_REGISTRY
#   3. add one line to MLFLOW_LOGGER_REGISTRY
#   4. add config in config.yaml
#   model_training.py is never touched

MODEL_REGISTRY = {
    "lightgbm"      : lgb.LGBMRegressor,
    "xgboost"   : xgb.XGBRegressor

}


# ── MLFLOW LOGGER REGISTRY ────────────────────────────
#
# Each model type has its own MLflow logging function
# lightgbm → mlflow.lightgbm.log_model
# xgboost  → mlflow.xgboost.log_model
# sklearn  → mlflow.sklearn.log_model
#
# model_training.py calls get_mlflow_logger(model_type)
# gets back the correct function
# calls it — no if/elif needed anywhere

MLFLOW_LOGGER_REGISTRY = {
    "lightgbm"      : mlflow.lightgbm.log_model,
    "xgboost"       : mlflow.xgboost.log_model
}


# ── EARLY STOPPING REGISTRY ───────────────────────────
#
# LightGBM early stopping → callbacks parameter
# XGBoost  early stopping → early_stopping_rounds parameter
# sklearn  models         → no early stopping
#
# model_training.py calls get_fit_kwargs(model_type, eval_set)
# gets back the correct kwargs for that model
# calls model.fit(X_train, y_train, **fit_kwargs)
# no if/elif needed anywhere

EARLY_STOPPING_REGISTRY = {
    "lightgbm" : {
        "callbacks" : [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200)
        ]
    },
    "xgboost"  : {
        "verbose"               : False
    },
    "random_forest" : {},
    "linear"        : {},
}


# ── PUBLIC FUNCTIONS ──────────────────────────────────

def get_model(model_type: str, params: dict):
    """
    Returns instantiated model from string name in YAML.

    Called by model_training.py as:
        model = get_model("lightgbm", {"n_estimators": 2000})

    Returns:
        LGBMRegressor(n_estimators=2000)

    model_training.py never imports lightgbm directly.
    It only knows get_model().
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type : '{model_type}'\n"
            f"Available types   : {list(MODEL_REGISTRY.keys())}\n"
            f"To add new type   : add one line to MODEL_REGISTRY "
            f"in model_factory.py"
        )

    model_class = MODEL_REGISTRY[model_type]
    return model_class(**params)


def get_mlflow_logger(model_type: str):
    """
    Returns correct MLflow logging function for model type.

    Called by model_training.py as:
        log_fn = get_mlflow_logger("lightgbm")
        log_fn(model, artifact_path="model", ...)

    Returns:
        mlflow.lightgbm.log_model

    Defaults to sklearn logger for unknown types.
    """
    return MLFLOW_LOGGER_REGISTRY.get(
        model_type,
        mlflow.sklearn.log_model
    )


def get_fit_kwargs(model_type: str, eval_set: list) -> dict:
    """
    Returns correct fit() keyword arguments for each model type.

    Called by model_training.py as:
        fit_kwargs = get_fit_kwargs("lightgbm", [(X_val, y_val)])
        model.fit(X_train, y_train, **fit_kwargs)

    For lightgbm returns:
        {
            "eval_set" : [(X_val, y_val)],
            "callbacks": [early_stopping(100), log_evaluation(200)]
        }

    For xgboost returns:
        {
            "eval_set"              : [(X_val, y_val)],
            "early_stopping_rounds" : 100,
            "verbose"               : False
        }

    For sklearn returns:
        {}   no early stopping
    """
    base_kwargs = EARLY_STOPPING_REGISTRY.get(model_type, {}).copy()

    if model_type in ("lightgbm", "xgboost"):
        base_kwargs["eval_set"] = eval_set

    return base_kwargs

