from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol


class StorageBackend(Protocol):
    def exists(self, path: str) -> bool: ...

    def list(self, path: str) -> list[str]: ...

    def open_binary(self, path: str) -> BinaryIO: ...


class LocalStorageBackend:
    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def list(self, path: str) -> list[str]:
        root = Path(path)
        if not root.exists():
            return []
        return sorted(str(child) for child in root.iterdir())

    def open_binary(self, path: str) -> BinaryIO:
        return Path(path).open("rb")
