"""Opt-in OpenAQ/Nominatim contract smoke check.

This script reads configuration through ``Settings.from_env``. It never prints
the API key or raw upstream response bodies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality_tool import get_air_quality  # noqa: E402


def main() -> int:
    location = sys.argv[1] if len(sys.argv) > 1 else "Berlin, Germany"
    result = get_air_quality(location)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in {
        "ok",
        "partial",
        "no_data",
        "ambiguous_location",
        "rejected",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
