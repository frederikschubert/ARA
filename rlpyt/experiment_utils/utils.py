import importlib
import pkgutil


def import_submodules(module):
    """Import all submodules of a module, recursively."""
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        module.__path__, module.__name__ + "."
    ):
        importlib.import_module(module_name)
