from models.registry import get_model, list_models, register_model

_MODEL_MODULES = [
    'models.lightgbm_model',
    'models.xgboost_model',
    'models.catboost_model',
    'models.random_forest_model',
    'models.logistic_regression_model',
    'models.linear_svc_model',
    'models.svc_model',
    'models.nu_svc_model',
]

for _mod in _MODEL_MODULES:
    try:
        __import__(_mod)
    except ImportError:
        pass
