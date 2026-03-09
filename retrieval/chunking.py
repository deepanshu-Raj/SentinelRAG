from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


def load_and_chunk_directory(data_dir: str) -> List[Chunk]:
    root = Path(data_dir)
    files = [p for p in root.rglob("*") if p.is_file()]

    all_chunks: List[Chunk] = []
    for file_path in files:
        rel_path = file_path.relative_to(root)
        text = read_text_file(file_path)
        pieces = chunk_text(text)

        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{rel_path.as_posix()}::chunk_{i}",
                    source=rel_path.as_posix(),
                    text=piece,
                )
            )

    return all_chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)


def chunk_from_dict(data: dict) -> Chunk:
    return Chunk(**data)
