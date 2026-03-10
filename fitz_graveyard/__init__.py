# fitz_graveyard/__init__.py
"""
fitz-graveyard: MCP server for local-first AI planning with fitz-ai integration.

Provides tools for analyzing projects, generating plans, and managing async planning jobs.
"""

import sys

if sys.platform == "win32":
    import platform

    # Workaround: platform._wmi_query() deadlocks on some Windows systems
    # (broken/slow WMI service). The openai and ollama SDKs call
    # platform.platform() / platform.machine() from background threads,
    # triggering the deadlock. Fix: replace _wmi_query with a no-op so the
    # WMI codepath is never reached. Registry and env-var fallbacks in
    # platform.py still provide correct values.
    if hasattr(platform, "_wmi_query"):
        def _wmi_disabled(*args, **kwargs):
            raise OSError("WMI disabled to prevent deadlock")
        platform._wmi_query = _wmi_disabled

__version__ = "0.2.0"
