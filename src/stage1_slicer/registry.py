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
            # Explicit imports for plugins inside get_plugin to fix auto-registration safely
            import src.stage1_slicer.plugins.field_slicer
            try:
                import src.stage1_slicer.plugins.math_model
            except ImportError:
                pass
            if name not in cls._plugins:
                raise ValueError(f"Plugin '{name}' not found. Available: {list(cls._plugins.keys())}")
        return cls._plugins[name]

    @classmethod
    def get_all_plugins(cls) -> Dict[str, Type[BaseSlicerPlugin]]:
        import src.stage1_slicer.plugins.field_slicer
        try:
            import src.stage1_slicer.plugins.math_model
        except ImportError:
            pass
        return cls._plugins

    @classmethod
    def discover_plugins(cls):
        """
        Explicitly imports built-in plugins instead of using pkgutil to avoid
        compile-time and packaging issues (e.g. PyInstaller, Windows).
        """
        import src.stage1_slicer.plugins.field_slicer

# Auto-discover on module import
PluginRegistry.discover_plugins()
