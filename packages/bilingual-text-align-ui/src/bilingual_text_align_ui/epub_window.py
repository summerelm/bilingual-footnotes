"""Tkinter presentation layer for bilingual EPUB creation."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import cast

from .build_application import EpubBuildRequest, EpubBuildResult, run_epub_build
from .storage_window import StorageWindow

PHASE_LABELS = {
    "preparing": "Preparing alignment worker",
    "reading": "Reading both EPUBs",
    "containment": "Locating corresponding text",
    "fine-alignment": "Aligning corresponding text",
    "rendering": "Adding translation footnotes",
    "verification": "Verifying the finished EPUB",
}


def suggested_output(reading: str) -> tuple[str | None, str]:
    """Suggest a nearby bilingual filename from the selected reading book."""
    if not reading.strip():
        return None, "bilingual.epub"
    path = Path(reading.strip())
    return str(path.parent), f"{path.stem}-bilingual.epub"


class EpubWindow:
    """A small, non-blocking desktop form for one EPUB build at a time."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._running = False
        self._resources_busy = False
        self._storage_window: StorageWindow | None = None
        self._started_at = 0.0
        self._phase_label = "Preparing EPUB build"
        self._base = tk.StringVar()
        self._translation = tk.StringVar()
        self._output = tk.StringVar()
        self._status = tk.StringVar(value="Choose two corresponding EPUB books.")

        root.title("Bilingual Footnotes")
        root.minsize(860, 540)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self._root, padding=(34, 30), style="App.TFrame")
        frame.grid(sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        header = ttk.Frame(frame, style="App.TFrame")
        header.grid(row=0, column=0, pady=(0, 22), sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Create a bilingual EPUB", style="Title.TLabel").grid(
            row=0, column=0, pady=(0, 5), sticky="w"
        )
        ttk.Label(
            header,
            text="Add popup translation notes to the book you want to read. Processing stays local.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w")
        self._storage_button = ttk.Button(
            header,
            text="Model & storage…",
            command=self._open_storage,
            style="Secondary.TButton",
        )
        self._storage_button.grid(row=0, column=1, rowspan=2, padx=(18, 0), sticky="e")

        ttk.Separator(frame, style="Rule.TSeparator").grid(row=1, column=0, sticky="ew")
        workflow = ttk.Frame(frame, style="App.TFrame")
        workflow.grid(row=2, column=0, sticky="nsew")
        workflow.columnconfigure(2, weight=1)
        self._path_row(
            workflow,
            0,
            "1",
            "Reading book",
            "The edition you want to read",
            self._base,
            self._choose_base,
        )
        self._path_row(
            workflow,
            2,
            "2",
            "Translation book",
            "The edition that supplies the notes",
            self._translation,
            self._choose_translation,
        )
        self._path_row(
            workflow,
            4,
            "3",
            "Save new EPUB as",
            "Your source books are protected from overwriting",
            self._output,
            self._choose_output,
        )

        self._progress = ttk.Progressbar(
            frame, mode="indeterminate", style="Accent.Horizontal.TProgressbar"
        )
        self._progress.grid(row=3, column=0, pady=(22, 12), sticky="ew")
        footer = ttk.Frame(frame, style="App.TFrame")
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self._status, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._run_button = ttk.Button(
            footer, text="Create EPUB", command=self._start, style="Primary.TButton"
        )
        self._run_button.grid(row=0, column=1, padx=(18, 0), sticky="e")

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        number: str,
        title: str,
        explanation: str,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        if row:
            ttk.Separator(parent, style="Rule.TSeparator").grid(
                row=row - 1, column=0, columnspan=4, sticky="ew"
            )
        ttk.Label(parent, text=number, style="StepNumber.TLabel").grid(
            row=row, column=0, padx=(12, 20), pady=24, sticky="nw"
        )
        description = ttk.Frame(parent, style="Row.TFrame")
        description.grid(row=row, column=1, padx=(0, 28), pady=24, sticky="nw")
        ttk.Label(description, text=title, style="StepTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(description, text=explanation, style="StepBody.TLabel").grid(
            row=1, column=0, pady=(4, 0), sticky="w"
        )
        ttk.Entry(parent, textvariable=variable, style="Field.TEntry").grid(
            row=row, column=2, pady=24, sticky="ew"
        )
        ttk.Button(parent, text="Choose…", command=command, style="Secondary.TButton").grid(
            row=row, column=3, padx=(12, 12), pady=24
        )

    def _choose_base(self) -> None:
        self._choose_input(self._base)

    def _choose_translation(self) -> None:
        self._choose_input(self._translation)

    def _choose_input(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            filetypes=(("EPUB books", "*.epub"), ("All files", "*"))
        )
        if selected:
            variable.set(selected)

    def _choose_output(self) -> None:
        initialdir, initialfile = suggested_output(self._base.get())
        selected = filedialog.asksaveasfilename(
            defaultextension=".epub",
            filetypes=(("EPUB books", "*.epub"), ("All files", "*")),
            confirmoverwrite=False,
            initialdir=initialdir,
            initialfile=initialfile,
        )
        if selected:
            self._output.set(selected)

    def _request(self) -> EpubBuildRequest:
        values = (
            self._base.get().strip(),
            self._translation.get().strip(),
            self._output.get().strip(),
        )
        if not all(values):
            raise ValueError("Choose the reading EPUB, translation EPUB, and output path.")
        return EpubBuildRequest(Path(values[0]), Path(values[1]), Path(values[2]))

    def _open_storage(self) -> None:
        if self._storage_window is not None:
            self._storage_window.focus()
            return
        self._storage_window = StorageWindow(
            self._root,
            available=not self._running,
            report_busy=self._set_resources_busy,
            on_close=self._storage_closed,
        )

    def _storage_closed(self) -> None:
        self._storage_window = None

    def _set_resources_busy(self, busy: bool) -> None:
        self._resources_busy = busy
        self._run_button.configure(state="disabled" if busy else "normal")

    def _start(self) -> None:
        if self._resources_busy:
            messagebox.showinfo(
                "Storage operation in progress",
                "Wait for the storage operation to finish before creating an EPUB.",
                parent=self._root,
            )
            return
        try:
            request = self._request()
        except ValueError as exc:
            messagebox.showerror("Cannot create EPUB", str(exc), parent=self._root)
            return
        self._running = True
        self._started_at = time.monotonic()
        self._run_button.configure(state="disabled")
        self._storage_button.configure(state="disabled")
        if self._storage_window is not None:
            self._storage_window.set_available(False)
        self._progress.configure(mode="indeterminate", value=0)
        self._progress.start(12)
        self._phase_label = "Preparing EPUB build"
        self._status.set(f"{self._phase_label}…")
        threading.Thread(target=self._work, args=(request,), daemon=True).start()
        self._root.after(200, self._poll)

    def _work(self, request: EpubBuildRequest) -> None:
        try:
            result = run_epub_build(
                request,
                report_progress=lambda phase: self._events.put(("progress", phase)),
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
            self._events.put(("error", str(exc)))
        else:
            self._events.put(("done", result))

    def _poll(self) -> None:
        try:
            kind, detail = self._events.get_nowait()
        except queue.Empty:
            elapsed = int(time.monotonic() - self._started_at)
            self._status.set(f"{self._phase_label}… {elapsed // 60}:{elapsed % 60:02d} elapsed")
            self._root.after(200, self._poll)
            return
        if kind == "progress":
            self._phase_label = PHASE_LABELS[cast(str, detail)]
            self._poll()
            return
        self._running = False
        self._progress.stop()
        self._run_button.configure(state="normal")
        self._storage_button.configure(state="normal")
        if self._storage_window is not None:
            self._storage_window.set_available(True)
        if kind == "done":
            self._progress.configure(mode="determinate", maximum=100, value=100)
            result = cast(EpubBuildResult, detail)
            self._status.set(
                f"Finished: {result.rendered_note_count} translation notes written to "
                f"{self._output.get()}"
            )
            messagebox.showinfo("EPUB complete", self._status.get(), parent=self._root)
        else:
            self._progress.configure(mode="determinate", maximum=100, value=0)
            self._status.set(f"EPUB build failed: {detail}")
            messagebox.showerror("EPUB build failed", str(detail), parent=self._root)

    def _close(self) -> None:
        if self._running or self._resources_busy:
            messagebox.showinfo(
                "Work in progress",
                "Wait for the current operation to finish before closing.",
                parent=self._root,
            )
            return
        self._root.destroy()
