agent_registry = {}


def register_agent(cls):
    agent_registry[cls.__name__] = cls
    return cls

