from pathlib import Path

from tsfm_lab.api import build_css


def test_the_native_module_the_css_compiler_leaves_in_temp_is_removed(tmp_path: Path, monkeypatch) -> None:
    root, temp = tmp_path / "root", tmp_path / "temp"
    (root / "web" / "src").mkdir(parents=True)
    (root / "web" / "src" / "app.css").write_text("@import 'tailwindcss';", encoding="utf-8")
    temp.mkdir()
    older = temp / ".1111-00000000.node"
    older.write_bytes(b"left by an earlier run")
    unrelated = temp / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    def fake_cli(command, **kwargs):
        (temp / ".2222-00000000.node").write_bytes(b"x" * 1024)  # what the real CLI leaves behind
        Path(command[command.index("-o") + 1]).write_text("/* css */", encoding="utf-8")

        class Done:
            returncode, stderr = 0, ""

        return Done()

    monkeypatch.setattr(build_css, "find_root", lambda: root)
    monkeypatch.setattr(build_css, "tailwind_binary", lambda: Path("tailwindcss"))
    monkeypatch.setattr(build_css.tempfile, "tempdir", str(temp))
    monkeypatch.setattr(build_css.subprocess, "run", fake_cli)

    assert build_css.ensure_css(force=True) is True
    assert not (temp / ".2222-00000000.node").exists()  # this run's leftover is gone
    assert older.exists() and unrelated.exists()  # nothing else is touched
