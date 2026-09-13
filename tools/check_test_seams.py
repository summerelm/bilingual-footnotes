"""Reject runtime patching in tests so dependencies stay explicit."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "setattr":
            found.append((node.lineno, "replace setattr patching with an injected dependency"))
        if isinstance(node, ast.ImportFrom) and node.module == "unittest.mock":
            if any(alias.name == "patch" for alias in node.names):
                found.append((node.lineno, "replace unittest.mock.patch with an injected dependency"))
        if isinstance(node, ast.Attribute) and node.attr == "patch":
            found.append((node.lineno, "replace mock.patch with an injected dependency"))
    return found


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tests = sorted((root / "packages").glob("*/tests/**/*.py"))
    failures = [
        (path, line, message)
        for path in tests
        for line, message in violations(path)
    ]
    for path, line, message in failures:
        print(f"{path.relative_to(root)}:{line}: {message}")
    if failures:
        return 1
    print(f"Test seams: {len(tests)} files use explicit dependencies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
