"""Strategy registry with auto-discovery.

Strategies placed in this package are auto-discovered when accessed.
Use ``get_strategy(name)`` to retrieve a strategy class by name.
"""

import importlib
import pkgutil
from pathlib import Path

from ..trade import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Register a strategy class. Use as a decorator on strategy classes."""
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[BaseStrategy]:
    """Get a strategy class by name.

    Triggers auto-discovery on first call.
    """
    _discover()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    _discover()
    return sorted(_REGISTRY.keys())


_discovered = False


def _discover() -> None:
    """Import all modules in this package to trigger @register decorators."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    pkg_path = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_path)]):
        if not info.name.startswith("_"):
            importlib.import_module(f".{info.name}", __package__)
