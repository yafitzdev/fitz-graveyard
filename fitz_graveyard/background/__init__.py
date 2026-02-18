# fitz_graveyard/background/__init__.py
"""
Background job processing system.

Exports:
    - BackgroundWorker: Sequential job processor
    - setup_signal_handlers: Graceful shutdown signal handling
    - ServerLifecycle: Startup recovery and shutdown coordination
"""

from fitz_graveyard.background.lifecycle import ServerLifecycle
from fitz_graveyard.background.signals import setup_signal_handlers
from fitz_graveyard.background.worker import BackgroundWorker

__all__ = ["BackgroundWorker", "setup_signal_handlers", "ServerLifecycle"]
