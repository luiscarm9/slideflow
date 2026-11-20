"""
This module is responsible for loading all the plugins that are installed in the system.
"""

import importlib.metadata
import sys

from slideflow.util import log


def _iter_plugin_entry_points():
    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 10) or hasattr(eps, 'select'):
        return eps.select(group='slideflow.plugins')
    return eps.get('slideflow.plugins', ())


def load_plugins():
    for entry_point in _iter_plugin_entry_points():
        try:
            register = entry_point.load()
        except Exception as e:
            log.warning(f"Failed to load plugin {entry_point.name}: {e}")
            continue
        if not callable(register):
            log.error(
                f"Plugin {entry_point.name} did not export a callable; "
                "skipping."
            )
            continue
        try:
            register()
        except Exception as e:
            log.warning(
                f"Plugin {entry_point.name} register() raised: {e}"
            )


load_plugins()
