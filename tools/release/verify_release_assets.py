"""Validate the complete set of desktop release archives."""

from __future__ import annotations

import argparse
import hashlib
import struct
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_version() -> str:
    package = ROOT / "packages/bilingual-text-align-ui/pyproject.toml"
    with package.open("rb") as source:
        return str(tomllib.load(source)["project"]["version"])


def expected_assets(version: str) -> dict[str, str]:
    return {
        f"Bilingual-Footnotes-{version}-macOS-arm64.dmg": "dmg",
        f"Bilingual-Footnotes-{version}-macOS-x86_64.dmg": "dmg",
        f"Bilingual-Footnotes-Setup-{version}-Windows-x86_64.exe": "windows-installer",
    }


def validate_checksum(asset: Path) -> None:
    checksum = asset.with_suffix(asset.suffix + ".sha256")
    with asset.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    expected = f"{digest}  {asset.name}\n".encode()
    if checksum.read_bytes() != expected:
        raise RuntimeError(f"invalid or non-portable checksum: {checksum.name}")


def validate_dmg(asset: Path) -> None:
    if asset.stat().st_size < 512:
        raise RuntimeError(f"invalid UDIF disk image: {asset.name}")
    with asset.open("rb") as source:
        source.seek(-512, 2)
        trailer = source.read(512)
    if not trailer.startswith(b"koly"):
        raise RuntimeError(f"invalid UDIF disk image: {asset.name}")


def validate_windows_installer(asset: Path) -> None:
    with asset.open("rb") as source:
        header = source.read(0x40)
        if header.startswith(b"MZ") and len(header) == 0x40:
            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            source.seek(pe_offset)
            if source.read(4) == b"PE\0\0":
                return
    raise RuntimeError(f"invalid Windows executable: {asset.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    expected = expected_assets(project_version())
    assets = [
        path
        for path in arguments.directory.rglob("*")
        if path.is_file() and path.suffix in {".dmg", ".exe"}
    ]
    actual = {path.name for path in assets}
    if len(actual) != len(assets):
        raise SystemExit("duplicate release asset names")
    if actual != set(expected):
        detail = ", ".join(sorted(actual)) or "none"
        raise SystemExit(f"unexpected release asset set: {detail}")
    paths = {path.name: path for path in assets}
    for name, kind in expected.items():
        asset = paths[name]
        validate_checksum(asset)
        if kind == "dmg":
            validate_dmg(asset)
        else:
            validate_windows_installer(asset)
        print(f"Validated {name}: {kind}")


if __name__ == "__main__":
    main()
