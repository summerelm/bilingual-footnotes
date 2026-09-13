"""Tkinter window for explicit model and processing-cache management."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from .resource_application import (
    ResourceOverview,
    clear_processing_cache,
    inspect_managed_resources,
    install_semantic_model,
    remove_semantic_model,
)


class StorageWindow:
    """Model and processing-cache controls."""

    def __init__(
        self,
        parent: tk.Tk,
        *,
        available: bool,
        report_busy: Callable[[bool], None],
        on_close: Callable[[], None],
    ) -> None:
        self._report_busy = report_busy
        self._on_close = on_close
        self._available = available
        self._busy = False
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._model_status = tk.StringVar()
        self._model_path = tk.StringVar()
        self._cache_status = tk.StringVar()
        self._cache_path = tk.StringVar()
        self._operation = tk.StringVar()
        self._window = tk.Toplevel(parent)
        self._window.title("Model & storage · Bilingual Footnotes")
        self._window.minsize(720, 500)
        self._window.transient(parent)
        self._window.protocol("WM_DELETE_WINDOW", self.close)
        self._buttons: list[ttk.Button] = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        frame = ttk.Frame(self._window, padding=(34, 30), style="App.TFrame")
        frame.grid(sticky="nsew")
        self._window.columnconfigure(0, weight=1)
        self._window.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Model & storage", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Keep the alignment model and reusable processing data under your control.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, pady=(4, 22), sticky="w")
        ttk.Separator(frame, orient="horizontal", style="Rule.TSeparator").grid(
            row=2, column=0, sticky="ew"
        )
        self._resource_card(
            frame,
            3,
            "Semantic model",
            "Required for alignment. Install it once; verify or repair it independently.",
            self._model_status,
            self._model_path,
            (
                ("Install / repair", self._install),
                ("Verify", self._verify),
                ("Remove", self._remove_model),
            ),
        )
        self._resource_card(
            frame,
            5,
            "Processing cache",
            "Optional book-derived embeddings. It grows as needed and can be cleared using Clear Cache.",
            self._cache_status,
            self._cache_path,
            (("Clear Cache", self._clear_cache),),
        )
        ttk.Separator(frame, orient="horizontal", style="Rule.TSeparator").grid(
            row=6, column=0, sticky="ew"
        )
        footer = ttk.Frame(frame, style="App.TFrame")
        footer.grid(row=7, column=0, pady=(18, 0), sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self._operation, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(footer, text="Done", command=self.close, style="Secondary.TButton").grid(
            row=0, column=1, sticky="e"
        )

    def _resource_card(
        self,
        parent: ttk.Frame,
        row: int,
        title: str,
        explanation: str,
        status: tk.StringVar,
        path: tk.StringVar,
        actions: tuple[tuple[str, Callable[[], None]], ...],
    ) -> None:
        if row > 3:
            ttk.Separator(parent, orient="horizontal", style="Rule.TSeparator").grid(
                row=row - 1, column=0, sticky="ew"
            )
        card = ttk.Frame(parent, padding=(0, 22), style="Card.TFrame")
        card.grid(row=row, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=status, style="CardStatus.TLabel").grid(
            row=0, column=1, sticky="e"
        )
        ttk.Label(card, text=explanation, style="CardBody.TLabel").grid(
            row=1, column=0, columnspan=2, pady=(7, 3), sticky="w"
        )
        ttk.Label(card, textvariable=path, style="Path.TLabel").grid(
            row=2, column=0, columnspan=2, pady=(0, 10), sticky="w"
        )
        action_bar = ttk.Frame(card, style="Card.TFrame")
        action_bar.grid(row=3, column=0, columnspan=2, sticky="w")
        for column, (label, command) in enumerate(actions):
            style = "Danger.TButton" if label in {"Remove", "Clear Cache"} else "Secondary.TButton"
            button = ttk.Button(action_bar, text=label, command=command, style=style)
            button.grid(row=0, column=column, padx=(0, 8))
            self._buttons.append(button)

    def refresh(self) -> None:
        self._show(inspect_managed_resources())
        self._set_button_state()

    def _show(self, overview: ResourceOverview) -> None:
        self._model_status.set(overview.model_status)
        self._model_path.set(str(overview.model_path.resolve()))
        self._cache_status.set(overview.cache_status)
        self._cache_path.set(str(overview.cache_path.resolve()))

    def set_available(self, available: bool) -> None:
        self._available = available
        self._set_button_state()

    def _set_button_state(self) -> None:
        state = "normal" if self._available and not self._busy else "disabled"
        for button in self._buttons:
            button.configure(state=state)

    def _install(self) -> None:
        self._start_operation("Installing or repairing semantic model…", install_semantic_model)

    def _verify(self) -> None:
        self._start_operation(
            "Verifying semantic model without network access…",
            lambda: install_semantic_model(verify_only=True),
        )

    def _remove_model(self) -> None:
        if messagebox.askyesno(
            "Remove semantic model?",
            "Future EPUB builds will require installing the model again.",
            parent=self._window,
        ):
            self._start_operation("Removing semantic model…", remove_semantic_model)

    def _clear_cache(self) -> None:
        if messagebox.askyesno(
            "Clear processing cache?",
            "This removes reusable embeddings. Your books and semantic model are not affected.",
            parent=self._window,
        ):
            self._start_operation("Clearing processing cache…", clear_processing_cache)

    def _start_operation(self, label: str, operation: Callable[[], object]) -> None:
        self._busy = True
        self._report_busy(True)
        self._operation.set(label)
        self._set_button_state()
        threading.Thread(target=self._work, args=(operation,), daemon=True).start()
        self._window.after(200, self._poll)

    def _work(self, operation: Callable[[], object]) -> None:
        try:
            operation()
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self._events.put(("error", str(exc)))
        else:
            self._events.put(("done", "Storage updated."))

    def _poll(self) -> None:
        try:
            kind, detail = self._events.get_nowait()
        except queue.Empty:
            self._window.after(200, self._poll)
            return
        self._busy = False
        self._report_busy(False)
        self.refresh()
        if kind == "error":
            self._operation.set(f"Storage operation failed: {detail}")
            messagebox.showerror("Storage operation failed", str(detail), parent=self._window)
        else:
            self._operation.set(str(detail))

    def focus(self) -> None:
        self._window.lift()
        self._window.focus_force()

    def close(self) -> None:
        if self._busy:
            messagebox.showinfo(
                "Storage operation in progress",
                "Wait for the storage operation to finish before closing.",
                parent=self._window,
            )
            return
        self._window.destroy()
        self._on_close()
