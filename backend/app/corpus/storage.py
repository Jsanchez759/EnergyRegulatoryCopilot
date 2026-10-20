import hashlib
from pathlib import Path
from typing import Protocol

from asyncer import asyncify


class BlobStorage(Protocol):
    async def put(self, data: bytes, suffix: str = "") -> str: ...

    async def get(self, key: str) -> bytes: ...


class LocalBlobStorage:
    """Content-addressed files on disk; keys stay valid when moving to an object store."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents:
            raise ValueError("Storage key escapes the storage root.")
        return path

    async def put(self, data: bytes, suffix: str = "") -> str:
        digest = hashlib.sha256(data).hexdigest()
        key = f"{digest[:2]}/{digest}{suffix.lower()}"
        path = self._path(key)
        if not path.exists():
            await asyncify(self._write)(path, data)
        return key

    async def get(self, key: str) -> bytes:
        return await asyncify(self._path(key).read_bytes)()

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
