algo_registry = {}


def register_algo(cls):
    algo_registry[cls.__name__] = cls
    return cls

