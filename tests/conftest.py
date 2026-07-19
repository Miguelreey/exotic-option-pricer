import sys
from pathlib import Path

# Repo root on sys.path so `exotic_option_pricer` imports resolve without
# an editable install (e.g. running pytest from a fresh clone).
sys.path.insert(0, str(Path(__file__).parent.parent))
