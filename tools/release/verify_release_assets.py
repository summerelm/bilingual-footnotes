"""Validate the complete set of desktop release archives."""

from __future__ import annotations

import argparse
import hashlib
import struct
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Bilingual Footnotes"


def project_version() -> str:
    package = ROOT / "packages/bilingual-text-align-ui/pyproject.toml"
    with package.open("rb") as source:
        return str(tomllib.load(source)["project"]["version"])


def expected_archives(version: str) -> dict[str, tuple[dict[str, str], str]]:
    mac_members = {
        "application": f"{APP_NAME}.app/Contents/MacOS/{APP_NAME}",
        "worker": (f"{APP_NAME}.app/Contents/Resources/semantic-worker/bilingual-align-worker"),
        "license": (f"{APP_NAME}.app/Contents/Resources/licenses/Bilingual-Footnotes-MIT.txt"),
    }
    windows_members = {
        "application": f"{APP_NAME}/{APP_NAME}.exe",
        "worker": f"{APP_NAME}/semantic-worker/bilingual-align-worker.exe",
        "license": f"{APP_NAME}/licenses/Bilingual-Footnotes-MIT.txt",
    }
    return {
        f"Bilingual-Footnotes-{version}-macOS-arm64.zip": (mac_members, "arm64"),
        f"Bilingual-Footnotes-{version}-macOS-x86_64.zip": (mac_members, "x86_64"),
        f"Bilingual-Footnotes-{version}-Windows-x86_64.zip": (windows_members, "windows-x86_64"),
    }


def validate_checksum(archive: Path) -> None:
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    expected = f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n".encode()
    if checksum.read_bytes() != expected:
        raise RuntimeError(f"invalid or non-portable checksum: {checksum.name}")


def executable_architecture(data: bytes) -> str:
    if data.startswith(b"\xcf\xfa\xed\xfe"):
        cpu_type = struct.unpack_from("<I", data, 4)[0]
        architectures = {0x01000007: "x86_64", 0x0100000C: "arm64"}
        if cpu_type in architectures:
            return architectures[cpu_type]
    if data.startswith(b"MZ") and len(data) >= 0x40:
        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe_offset : pe_offset + 4] == b"PE\0\0":
            machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
            if machine == 0x8664:
                return "windows-x86_64"
    raise RuntimeError("unrecognized executable architecture")


def validate_archive(
    archive: Path, required_members: dict[str, str], expected_architecture: str
) -> None:
    validate_checksum(archive)
    with zipfile.ZipFile(archive) as package:
        if corrupt := package.testzip():
            raise RuntimeError(f"corrupt ZIP member in {archive.name}: {corrupt}")
        missing = set(required_members.values()).difference(package.namelist())
        if missing:
            raise RuntimeError(f"missing from {archive.name}: {', '.join(sorted(missing))}")
        for kind in ("application", "worker"):
            actual = executable_architecture(package.read(required_members[kind]))
            if actual != expected_architecture:
                raise RuntimeError(
                    f"{archive.name} {kind} is {actual}, expected {expected_architecture}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    expected = expected_archives(project_version())
    archives = list(arguments.directory.rglob("*.zip"))
    actual = {path.name for path in archives}
    if len(actual) != len(archives):
        raise SystemExit("duplicate release archive names")
    if actual != set(expected):
        detail = ", ".join(sorted(actual)) or "none"
        raise SystemExit(f"unexpected release archive set: {detail}")
    paths = {path.name: path for path in archives}
    for name, (members, architecture) in expected.items():
        validate_archive(paths[name], members, architecture)
        print(f"Validated {name}: {architecture}")


if __name__ == "__main__":
    main()
