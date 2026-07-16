"""Print the octree builder source being imported by this Python environment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import octree_graph.cli as cli
import octree_graph.octree as octree


def _source_info(path: str) -> dict[str, str]:
    source = Path(path).resolve()
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = ""
    return {
        "path": str(source),
        "sha256_16": digest,
    }


def _octree_enforcement_info(path: str) -> dict[str, str]:
    source = Path(path).resolve()
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return {
        "has_mandatory_max_size_patch": str("mandatory_max_size_subdivision" in text),
        "has_old_budget_block_branch": str('block_reason = "max_leaf_cells"' in text),
    }


def main() -> None:
    print("cli:")
    for key, value in _source_info(cli.__file__ or "").items():
        print(f"  {key}: {value}")
    print("octree:")
    for key, value in _source_info(octree.__file__ or "").items():
        print(f"  {key}: {value}")
    for key, value in _octree_enforcement_info(octree.__file__ or "").items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
