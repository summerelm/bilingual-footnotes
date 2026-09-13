from bilingual_text_align_ui.theme import system_prefers_dark


def test_linux_appearance_uses_common_desktop_environment() -> None:
    assert system_prefers_dark("linux", {"GTK_THEME": "Adwaita:dark"})
    assert system_prefers_dark("linux", {"COLORFGBG": "15;0"})
    assert not system_prefers_dark("linux", {"GTK_THEME": "Adwaita"})
