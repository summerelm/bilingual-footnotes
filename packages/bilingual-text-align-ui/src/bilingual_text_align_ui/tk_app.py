"""Composition root for the Tkinter desktop application."""

from __future__ import annotations

import argparse
import ctypes
import sys
import tkinter as tk
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path

from .epub_window import EpubWindow
from .theme import apply_theme, system_prefers_dark


def _set_macos_application_icon(path: Path) -> None:
    """Set the running app's Dock icon, which Tk otherwise replaces with Python's."""
    if sys.platform != "darwin":
        return

    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    message_address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if message_address is None:
        return

    send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(message_address)
    send_object = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(message_address)
    send_string = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
    )(message_address)
    send_void_object = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        message_address
    )

    application = send(
        objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication")
    )
    ns_path = send_string(
        objc.objc_getClass(b"NSString"),
        objc.sel_registerName(b"stringWithUTF8String:"),
        str(path).encode(),
    )
    image = send_object(
        send(objc.objc_getClass(b"NSImage"), objc.sel_registerName(b"alloc")),
        objc.sel_registerName(b"initWithContentsOfFile:"),
        ns_path,
    )
    if application and image:
        send_void_object(application, objc.sel_registerName(b"setApplicationIconImage:"), image)


def _apply_application_icon(root: tk.Tk) -> None:
    icon = files("bilingual_text_align_ui").joinpath("assets/icon.png")
    with as_file(icon) as icon_path:
        image = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, image)
        root._bilingual_footnotes_icon = image  # type: ignore[attr-defined]
        _set_macos_application_icon(icon_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilingual-align-ui",
        description=(
            "Open the Bilingual Footnotes desktop app. Choose a reading EPUB, its\n"
            "corresponding translation, and a new output path; processing stays local."
        ),
        epilog=(
            "The packaged desktop app includes its semantic worker. Use Model & storage\n"
            "inside the app to install or verify the model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('bilingual-text-align-ui')}"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"Could not start the Tkinter UI: {exc}") from exc
    _apply_application_icon(root)
    apply_theme(root, dark=system_prefers_dark())
    EpubWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
