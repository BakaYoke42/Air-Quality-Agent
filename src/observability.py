"""Optional Langfuse tracing that never becomes an agent dependency.

The project remains runnable when Langfuse is not installed or its keys are
absent.  When configured, observations nest through Langfuse's current-context
API and are explicitly flushed before the short-lived CLI process exits.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _enabled_value(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass
class _NoopObservation:
    """Small stand-in with the same update operation used by the agent."""

    def update(self, **_: Any) -> None:
        return None


class Observability:
    """Failure-isolated wrapper around the optional Langfuse Python SDK."""

    def __init__(self, client: Any | None, reason: str = "") -> None:
        self.client = client
        self.reason = reason

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @classmethod
    def from_environment(cls) -> "Observability":
        requested = _enabled_value(os.getenv("LANGFUSE_ENABLED", "true"))
        if not requested:
            return cls(None, "disabled by LANGFUSE_ENABLED")

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if not public_key or not secret_key:
            return cls(None, "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not configured")

        try:
            from langfuse import get_client
        except ImportError:
            return cls(None, "the 'langfuse' package is not installed")

        try:
            return cls(get_client())
        except Exception as exc:  # Observability must never break the agent.
            return cls(None, f"Langfuse initialisation failed: {type(exc).__name__}")

    @contextmanager
    def observation(
        self,
        *,
        as_type: str,
        name: str,
        input: Any | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled:
            yield _NoopObservation()
            return

        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
        }
        if input is not None:
            kwargs["input"] = input
        if model:
            kwargs["model"] = model
        if metadata:
            kwargs["metadata"] = metadata

        try:
            manager = self.client.start_as_current_observation(**kwargs)
        except Exception as exc:
            print(
                f"[trace] Langfuse span setup failed ({type(exc).__name__}); continuing.",
                file=sys.stderr,
                flush=True,
            )
            yield _NoopObservation()
            return

        # Do not catch errors raised by the actual agent body.
        with manager as observation:
            yield observation

    @staticmethod
    def update(observation: Any, **kwargs: Any) -> None:
        try:
            observation.update(**kwargs)
        except Exception as exc:
            print(
                f"[trace] Langfuse update failed ({type(exc).__name__}); continuing.",
                file=sys.stderr,
                flush=True,
            )

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self.client.flush()
        except Exception as exc:
            print(
                f"[trace] Langfuse flush failed ({type(exc).__name__}); continuing.",
                file=sys.stderr,
                flush=True,
            )

    def auth_check(self) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(self.client.auth_check())
        except Exception:
            return False

    def status(self) -> str:
        return "enabled" if self.enabled else f"disabled ({self.reason})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check optional Langfuse configuration.")
    parser.add_argument("--check", action="store_true", help="Validate configured credentials.")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    observer = Observability.from_environment()
    print(f"Langfuse: {observer.status()}")
    if args.check and observer.enabled:
        valid = observer.auth_check()
        print("Authentication: OK" if valid else "Authentication: FAILED")
        return 0 if valid else 1
    return 0 if observer.enabled or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Observability"]
