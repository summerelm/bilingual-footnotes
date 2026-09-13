"""Create the pinned Python 3.12 Vecalign + LaBSE worker environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path

from .resources import default_model_storage, prepare_model
from .vecalign_labse import DEFAULT_MODEL, DEFAULT_MODEL_REVISION

VECALIGN_COMMIT = "f37262758955133d0c9ef1fdff45eba25842a62c"
PACKAGES = (
    "numpy==1.26.4",
    "Cython==3.2.9",
    "torch==2.2.2",
    "scipy==1.13.1",
    "scikit-learn==1.5.2",
    "transformers==4.57.6",
    "sentence-transformers==5.7.0",
    f"vecalign @ git+https://github.com/thompsonb/vecalign.git@{VECALIGN_COMMIT}",
)


def build_parser() -> argparse.ArgumentParser:
    default_environment = Path(".venv-bilingual")
    default_python = _environment_python(default_environment)
    parser = argparse.ArgumentParser(
        prog="bilingual-align-setup",
        description=(
            "Prepare the local semantic worker used by bilingual-align and\n"
            "bilingual-footnotes.\n\n"
            "Setup creates a Python 3.12 environment, installs the pinned alignment stack,\n"
            "and downloads the semantic model. It requires uv, network access, and several\n"
            "gigabytes of free space."
        ),
        epilog=(
            "example:\n"
            "  bilingual-align-setup --python python3.12 --venv .venv-bilingual\n\n"
            f"After setup, the processing commands use {default_python} by default.\n"
            "If you choose another location, pass its Python with --aligner-python."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('bilingual-text-align')}"
    )
    parser.add_argument(
        "--python",
        default="python3.12",
        metavar="PYTHON",
        help="Python 3.12 executable used to create the worker (default: python3.12)",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=default_environment,
        metavar="DIRECTORY",
        help="create the worker environment here (default: .venv-bilingual)",
    )
    parser.add_argument(
        "--model-storage",
        type=Path,
        default=default_model_storage(),
        metavar="DIRECTORY",
        help="install the semantic model here (default: OS user cache)",
    )
    return parser


def _run(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _environment_python(environment: Path, platform: str = os.name) -> Path:
    return environment / ("Scripts/python.exe" if platform == "nt" else "bin/python")


def _run_setup_step(
    command: Sequence[str],
    failure: str,
    *,
    capture_output: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(command, capture_output=capture_output)
    except FileNotFoundError as exc:
        raise SystemExit(f"{failure}: command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail.splitlines()[-1]}" if detail else ""
        raise SystemExit(f"{failure}{suffix}") from exc


def main(
    argv: list[str] | None = None,
    *,
    locator: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    model_preparer: Callable[..., None] = prepare_model,
) -> None:
    args = build_parser().parse_args(argv)
    uv = locator("uv")
    if uv is None:
        raise SystemExit("uv is required; install it before setting up the aligner")
    selected = locator(args.python)
    if selected is None:
        raise SystemExit(
            f"Python 3.12 executable not found: {args.python}. "
            "Install Python 3.12 or pass its path with --python."
        )
    version = _run_setup_step(
        [selected, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        "could not check the selected Python version",
        capture_output=True,
        runner=runner,
    ).stdout.strip()
    if version != "3.12":
        raise SystemExit(f"the tested semantic stack requires Python 3.12; got {version}")
    environment = args.venv.resolve()
    _run_setup_step(
        [uv, "venv", "--python", selected, str(environment)],
        "could not create the semantic worker environment",
        runner=runner,
    )
    python = _environment_python(environment)
    _run_setup_step(
        [uv, "pip", "install", "--python", str(python), *PACKAGES],
        "could not install the semantic worker dependencies; check the network output above",
        runner=runner,
    )
    _run_setup_step(
        [str(python), "-c", "import numpy, torch, vecalign, sentence_transformers"],
        "the semantic worker was installed but its packages could not be imported",
        runner=runner,
    )
    try:
        model_preparer(
            python,
            Path(__file__).with_name("worker.py"),
            DEFAULT_MODEL,
            DEFAULT_MODEL_REVISION,
            args.model_storage,
        )
    except RuntimeError as exc:
        raise SystemExit(f"the semantic worker was installed but the model was not: {exc}") from exc
    print(f"Vecalign worker created at {python}")
    print(f"Semantic model installed at {args.model_storage.resolve()}")


if __name__ == "__main__":
    main()
