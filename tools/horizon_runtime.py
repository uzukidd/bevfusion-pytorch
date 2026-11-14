"""Process-start compatibility for the bundled Horizon plugin on newer glibc."""
from __future__ import annotations

import os
import sys


TUNABLE = "glibc.rtld.execstack"


def loader_environment(environment):
    """Preserve other loader settings while selecting the plugin-compatible value."""
    result = dict(environment)
    parts = [part for part in result.get("GLIBC_TUNABLES", "").split(":")
             if part and not part.startswith(TUNABLE + "=")]
    result["GLIBC_TUNABLES"] = ":".join(parts + [TUNABLE + "=2"])
    return result


def ensure_plugin_loader():
    """Restart Linux CLI before torch/plugin imports; dlopen reads startup settings."""
    if sys.platform != "linux":
        return
    environment = loader_environment(os.environ)
    if environment["GLIBC_TUNABLES"] == os.environ.get("GLIBC_TUNABLES", ""):
        return
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
