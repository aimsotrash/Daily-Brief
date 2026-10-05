"""Allow ``python -m daily_brief``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
