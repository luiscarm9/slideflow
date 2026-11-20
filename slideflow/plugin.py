"""
This module is responsible for loading all the plugins that are installed in the system.
"""

import importlib.metadata
import sys


def _iter_plugin_entry_points():
    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 10) or hasattr(eps, 'select'):
        return eps.select(group='slideflow.plugins')
    return eps.get('slideflow.plugins', ())


def load_plugins():
    for entry_point in _iter_plugin_entry_points():
        register = entry_point.load()
        register()


load_plugins()
