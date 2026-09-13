from __future__ import annotations

import threading
from time import monotonic

import pytest

from bilingual_text_align.progress import ElapsedHeartbeat


def test_heartbeat_reports_label_and_elapsed_time() -> None:
    messages: list[str] = []
    with ElapsedHeartbeat(lambda: "fine-alignment", 0.001, messages.append):
        deadline = monotonic() + 1
        while not messages and monotonic() < deadline:
            threading.Event().wait(0.001)

    assert messages
    assert messages[0].startswith("[fine-alignment] Still working;")
    assert messages[0].endswith("s elapsed")


def test_heartbeat_can_be_disabled() -> None:
    messages: list[str] = []
    with ElapsedHeartbeat(lambda: "alignment", 0, messages.append):
        pass
    assert messages == []


def test_heartbeat_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="zero or positive"):
        ElapsedHeartbeat(lambda: "alignment", -1)
