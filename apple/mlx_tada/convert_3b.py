import sys
from pathlib import Path

from .convert import convert_all

if len(sys.argv) != 2:
    print("Usage: python -m mlx_tada.convert_3b <output_dir>")
    sys.exit(1)

convert_all("HumeAI/tada-3b-ml", "HumeAI/tada-codec", Path(sys.argv[1]))
