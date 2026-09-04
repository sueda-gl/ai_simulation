# src/engine/log.py
"""Console-only prefixed logging for the engine (the legacy '[Copula] ...' lines)."""


class EngineLog:
    """
    Callable that prints ``"<prefix> <message>"`` to stdout.

    The engine's progress lines are console-only (nothing parses them), so a plain
    ``print`` with the mode's prefix is all that is needed.
    """

    __slots__ = ("prefix",)

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, message: str) -> None:
        print(f"{self.prefix} {message}")

    def __repr__(self) -> str:
        return f"EngineLog({self.prefix!r})"
