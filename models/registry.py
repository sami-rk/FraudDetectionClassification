_registry = {}


def register_model(name, cls):
    _registry[name] = cls


def get_model(name, params=None):
    if name not in _registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(_registry.keys())}")
    return _registry[name](params=params)


def list_models():
    return list(_registry.keys())
