from pkgutil import extend_path

# Resolve installed extensions even when this local source package is imported.
__path__ = extend_path(__path__, __name__)

from .bev_pool import bev_pool
