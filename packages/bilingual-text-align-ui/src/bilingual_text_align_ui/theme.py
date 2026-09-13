"""Site-derived color palettes and Tkinter theme configuration."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk


@dataclass(frozen=True)
class Palette:
    paper: str
    surface: str
    ink: str
    soft_ink: str
    forest: str
    sage: str
    vermilion: str
    gold: str
    rule: str
    button_ink: str


LIGHT = Palette(
    paper="#f6f0e4",
    surface="#fffaf0",
    ink="#2b1712",
    soft_ink="#6d5a50",
    forest="#3f4928",
    sage="#e8e3cf",
    vermilion="#922f26",
    gold="#b28a4b",
    rule="#cbbdad",
    button_ink="#ffffff",
)

DARK = Palette(
    paper="#1d1511",
    surface="#291e18",
    ink="#f4ead8",
    soft_ink="#c4b3a4",
    forest="#aab28a",
    sage="#343323",
    vermilion="#d36a58",
    gold="#c69e5d",
    rule="#57483e",
    button_ink="#1d1511",
)


def system_prefers_dark(
    platform: str = sys.platform, environment: Mapping[str, str] | None = None
) -> bool:
    """Read the host appearance without requiring a UI preference toggle."""
    values = os.environ if environment is None else environment
    if platform == "darwin":
        try:
            result = subprocess.run(  # noqa: S603
                ("/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"),
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.stdout.strip().lower() == "dark"
    if platform == "win32":
        registry = Path(values.get("SystemRoot", r"C:\Windows")) / "System32" / "reg.exe"
        try:
            result = subprocess.run(  # noqa: S603
                (
                    str(registry),
                    "query",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                    "/v",
                    "AppsUseLightTheme",
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.rstrip().endswith("0x0")
    gtk_theme = values.get("GTK_THEME", "")
    if "dark" in gtk_theme.lower():
        return True
    color_scheme = values.get("COLORFGBG", "").rsplit(";", 1)[-1]
    return color_scheme.isdigit() and int(color_scheme) < 8


def apply_theme(root: tk.Tk, *, dark: bool) -> Palette:
    """Apply the site palette to the window and return the selected colors."""
    palette = DARK if dark else LIGHT
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=palette.paper)

    style.configure(".", font="TkDefaultFont", background=palette.paper, foreground=palette.ink)
    style.configure("App.TFrame", background=palette.paper)
    style.configure("Card.TFrame", background=palette.paper)
    style.configure("Row.TFrame", background=palette.paper)
    style.configure("App.TLabel", background=palette.paper, foreground=palette.ink)
    style.configure(
        "Eyebrow.TLabel",
        background=palette.paper,
        foreground=palette.vermilion,
        font=("TkDefaultFont", 9, "bold"),
    )
    style.configure(
        "Title.TLabel",
        background=palette.paper,
        foreground=palette.ink,
        font=("Georgia", 28, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=palette.paper,
        foreground=palette.soft_ink,
        font=("TkDefaultFont", 12),
    )
    style.configure(
        "StepNumber.TLabel",
        background=palette.paper,
        foreground=palette.forest,
        font=("Georgia", 22, "bold"),
    )
    style.configure(
        "StepTitle.TLabel",
        background=palette.paper,
        foreground=palette.ink,
        font=("Georgia", 15, "bold"),
    )
    style.configure(
        "StepBody.TLabel",
        background=palette.paper,
        foreground=palette.soft_ink,
        font=("TkDefaultFont", 10),
    )
    style.configure("Rule.TSeparator", background=palette.rule)
    style.configure(
        "CardTitle.TLabel",
        background=palette.paper,
        foreground=palette.ink,
        font=("TkDefaultFont", 11, "bold"),
    )
    style.configure("CardStatus.TLabel", background=palette.paper, foreground=palette.vermilion)
    style.configure("CardBody.TLabel", background=palette.paper, foreground=palette.soft_ink)
    style.configure(
        "Path.TLabel",
        background=palette.paper,
        foreground=palette.soft_ink,
        font=("TkFixedFont", 9),
    )
    style.configure(
        "Field.TLabel",
        background=palette.surface,
        foreground=palette.soft_ink,
        font=("TkDefaultFont", 9, "bold"),
    )
    style.configure("Status.TLabel", background=palette.paper, foreground=palette.soft_ink)
    style.configure(
        "Field.TEntry",
        fieldbackground=palette.paper,
        foreground=palette.ink,
        insertcolor=palette.ink,
        bordercolor=palette.rule,
        lightcolor=palette.rule,
        darkcolor=palette.rule,
        padding=9,
    )
    style.map(
        "Field.TEntry",
        bordercolor=[("focus", palette.vermilion)],
        lightcolor=[("focus", palette.vermilion)],
        darkcolor=[("focus", palette.vermilion)],
    )
    style.configure(
        "Secondary.TButton",
        background=palette.paper,
        foreground=palette.ink,
        bordercolor=palette.rule,
        lightcolor=palette.rule,
        darkcolor=palette.rule,
        padding=(12, 8),
        font=("TkDefaultFont", 10),
    )
    style.map(
        "Secondary.TButton",
        background=[("active", palette.sage), ("pressed", palette.sage)],
        foreground=[("active", palette.ink)],
    )
    style.configure(
        "Primary.TButton",
        background=palette.forest,
        foreground=palette.button_ink,
        borderwidth=0,
        padding=(20, 10),
        font=("TkDefaultFont", 10, "bold"),
    )
    style.configure(
        "Danger.TButton",
        background=palette.paper,
        foreground=palette.vermilion,
        borderwidth=0,
        padding=(10, 8),
        font=("TkDefaultFont", 10),
    )
    style.map(
        "Danger.TButton",
        background=[("active", palette.sage), ("pressed", palette.sage)],
        foreground=[("active", palette.vermilion)],
    )
    style.map(
        "Primary.TButton",
        background=[("active", palette.vermilion), ("pressed", palette.vermilion)],
        foreground=[("active", "#ffffff"), ("disabled", palette.soft_ink)],
    )
    style.configure(
        "Accent.Horizontal.TProgressbar",
        background=palette.vermilion,
        troughcolor=palette.sage,
        bordercolor=palette.sage,
        lightcolor=palette.vermilion,
        darkcolor=palette.vermilion,
    )
    return palette
