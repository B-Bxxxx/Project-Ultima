import importlib
import pkgutil
from typing import Dict, Type
import src.stage1_slicer.plugins
from src.stage1_slicer.base import BaseSlicerPlugin

class PluginRegistry:
    """
    Registry for dynamic loading of Stage 1 Slicer Plugins.
    """
    _plugins: Dict[str, Type[BaseSlicerPlugin]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(plugin_cls: Type[BaseSlicerPlugin]):
            cls._plugins[name] = plugin_cls
            return plugin_cls
        return wrapper

    @classmethod
    def get_plugin(cls, name: str) -> Type[BaseSlicerPlugin]:
        if name not in cls._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {list(cls._plugins.keys())}")
        return cls._plugins[name]

    @classmethod
    def get_all_plugins(cls) -> Dict[str, Type[BaseSlicerPlugin]]:
        return cls._plugins

    @classmethod
    def discover_plugins(cls):
        """
        Dynamically load all modules in the plugins package so they register themselves.
        """
        package = src.stage1_slicer.plugins
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package.__name__}.{module_name}")

# Auto-discover on module import
PluginRegistry.discover_plugins()
