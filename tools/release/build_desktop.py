"""Build and verify one native Bilingual Footnotes desktop archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ICON = ROOT / "packages/bilingual-text-align-ui/src/bilingual_text_align_ui/assets/icon.png"
APP_NAME = "Bilingual Footnotes"
WORKER_NAME = "bilingual-align-worker"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603


def project_version() -> str:
    package = ROOT / "packages/bilingual-text-align-ui/pyproject.toml"
    with package.open("rb") as source:
        return str(tomllib.load(source)["project"]["version"])


def pyinstaller(*arguments: str) -> None:
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", *arguments)


def create_windows_icon(target: Path) -> None:
    from PIL import Image

    with Image.open(ICON) as image:
        image.save(target, sizes=[(16, 16), (32, 32), (48, 48), (128, 128), (256, 256)])


def create_macos_icon(target: Path) -> None:
    from PIL import Image

    with Image.open(ICON) as image:
        image.save(target, format="ICNS")


def build_worker(dist: Path, work: Path) -> Path:
    pyinstaller(
        "--distpath",
        str(dist),
        "--workpath",
        str(work / "worker"),
        "--specpath",
        str(work),
        "--onedir",
        "--name",
        WORKER_NAME,
        "--exclude-module",
        "tkinter",
        "--collect-all",
        "sentence_transformers",
        "--collect-all",
        "vecalign",
        "--hidden-import",
        "transformers.models.bert.modeling_bert",
        "--hidden-import",
        "transformers.models.bert.tokenization_bert_fast",
        str(ROOT / "tools/release/worker_entry.py"),
    )
    return dist / WORKER_NAME


def build_app(dist: Path, work: Path, icon: Path) -> Path:
    arguments = [
        "--distpath",
        str(dist),
        "--workpath",
        str(work / "app"),
        "--specpath",
        str(work),
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--icon",
        str(icon),
        "--copy-metadata",
        "bilingual-text-align-ui",
        "--exclude-module",
        "sentence_transformers",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "sklearn",
        "--exclude-module",
        "scipy",
    ]
    if sys.platform == "darwin":
        arguments.extend(("--osx-bundle-identifier", "io.github.summerelm.bilingual-footnotes"))
    arguments.append(str(ROOT / "tools/release/app_entry.py"))
    pyinstaller(*arguments)
    return dist / (f"{APP_NAME}.app" if sys.platform == "darwin" else APP_NAME)


def copy_worker(app: Path, worker: Path) -> tuple[Path, Path]:
    if sys.platform == "darwin":
        destination = app / "Contents/Resources/semantic-worker"
        executable = destination / WORKER_NAME
        app_executable = app / f"Contents/MacOS/{APP_NAME}"
    else:
        destination = app / "semantic-worker"
        executable = destination / f"{WORKER_NAME}.exe"
        app_executable = app / f"{APP_NAME}.exe"
    shutil.copytree(worker, destination)
    return app_executable, executable


def copy_licenses(app: Path) -> None:
    resources = app / "Contents/Resources" if sys.platform == "darwin" else app
    destination = resources / "licenses"
    destination.mkdir()
    shutil.copy2(ROOT / "LICENSE", destination / "Bilingual-Footnotes-MIT.txt")
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "dependency").replace("/", "-")
        for installed_file in distribution.files or ():
            parts = tuple(part.lower() for part in installed_file.parts)
            filename = installed_file.name.lower()
            is_license = "licenses" in parts or filename.startswith(
                ("license", "copying", "notice")
            )
            source = installed_file.locate()
            if is_license and source.is_file():
                target = destination / f"{name}-{installed_file.name}"
                if not target.exists():
                    shutil.copy2(source, target)


def set_macos_metadata(app: Path, version: str) -> None:
    path = app / "Contents/Info.plist"
    with path.open("rb") as source:
        metadata = plistlib.load(source)
    metadata.update(
        CFBundleDisplayName=APP_NAME,
        CFBundleName=APP_NAME,
        CFBundleShortVersionString=version,
        CFBundleVersion=version,
    )
    with path.open("wb") as destination:
        plistlib.dump(metadata, destination)


def sign_macos_bundle(app: Path) -> None:
    """Restore a valid ad-hoc signature after adding the bundled worker and licenses."""

    run("codesign", "--force", "--deep", "--sign", "-", str(app))
    run("codesign", "--verify", "--deep", "--strict", str(app))


def verify(app_executable: Path, worker_executable: Path, version: str) -> None:
    app_result = subprocess.run(  # noqa: S603
        (str(app_executable), "--version"), capture_output=True, check=True, text=True
    )
    if sys.platform != "win32" and version not in app_result.stdout:
        raise RuntimeError(f"desktop version check failed: {app_result.stdout.strip()}")
    subprocess.run(  # noqa: S603
        (str(worker_executable), "--help"), capture_output=True, check=True, text=True
    )


def archive(app: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), str(output))
        return
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as destination:
        for path in app.rglob("*"):
            if path.is_file():
                destination.write(path, path.relative_to(app.parent))


def write_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    line = f"{digest}  {path.name}\n".encode("ascii")
    path.with_suffix(path.suffix + ".sha256").write_bytes(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("x86_64", "arm64"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/release")
    arguments = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("desktop releases must be built with Python 3.12")
    if sys.platform not in {"darwin", "win32"}:
        raise SystemExit("desktop releases are supported on macOS and Windows")
    machine = platform.machine().lower()
    actual_architecture = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
    if arguments.architecture != actual_architecture:
        raise SystemExit(
            f"requested {arguments.architecture}, but this runner is {actual_architecture}"
        )

    version = project_version()
    if (tag := os.environ.get("GITHUB_REF_NAME", "")).startswith("v") and tag[1:] != version:
        raise SystemExit(f"release tag {tag} does not match project version {version}")
    scratch = ROOT / "build/desktop"
    shutil.rmtree(scratch, ignore_errors=True)
    dist, work = scratch / "dist", scratch / "work"
    dist.mkdir(parents=True)
    icon = scratch / (
        "BilingualFootnotes.icns" if sys.platform == "darwin" else "BilingualFootnotes.ico"
    )
    if sys.platform == "darwin":
        create_macos_icon(icon)
    else:
        create_windows_icon(icon)

    worker = build_worker(dist, work)
    app = build_app(dist, work, icon)
    app_executable, worker_executable = copy_worker(app, worker)
    copy_licenses(app)
    if sys.platform == "darwin":
        set_macos_metadata(app, version)
        sign_macos_bundle(app)
    verify(app_executable, worker_executable, version)

    operating_system = "macOS" if sys.platform == "darwin" else "Windows"
    output = (
        arguments.output_dir.resolve()
        / f"Bilingual-Footnotes-{version}-{operating_system}-{arguments.architecture}.zip"
    )
    archive(app, output)
    write_checksum(output)
    print(output)


if __name__ == "__main__":
    main()
