"""Backwards-compatibility shim.

`import driftshield` still works after the rename to `driftshield_mini`.
New code should use `driftshield_mini` directly.
"""

import warnings

warnings.warn(
    "The 'driftshield' package was renamed to 'driftshield_mini'. "
    "Please update your imports: from driftshield_mini import DriftMonitor",
    DeprecationWarning,
    stacklevel=2,
)

from driftshield_mini import *
from driftshield_mini import (  # noqa: F401
    BaselineStats,
    DetectorType,
    DriftEvent,
    DriftMonitor,
    Severity,
    TraceEvent,
)
