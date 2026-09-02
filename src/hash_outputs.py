"""Print SHA-256 metadata for output files."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create JSON SHA-256 metadata for one or more files."
    )
    parser.add_argument("files", nargs="+", type=Path)
    arguments = parser.parse_args()

    metadata = []
    for path in arguments.files:
        if not path.is_file():
            parser.error(f"File not found: {path}")
        metadata.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()