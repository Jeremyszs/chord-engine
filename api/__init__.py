"""API layer for chord-engine."""

import time

# Shared app start time — set by api/main.py on import, read by
# api/routes/health.py for uptime calculation.  Stored here to avoid
# a circular import between main.py and routes/health.py.
_app_start_time: float = time.time()
