"""Module entry point; guarded so multiprocessing workers can import it safely."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
