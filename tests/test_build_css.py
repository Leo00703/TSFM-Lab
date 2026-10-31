from pathlib import Path

from tsfm_lab.api import build_css


def test_the_native_module_the_css_compiler_unpacks_goes_away_with_its_own_temp_folder(tmp_path: Path, monkeypatch) -> None:
    root, ambient = tmp_path / "root", tmp_path / "ambient"
    (root / "web" / "src").mkdir(parents=True)
    (root / "web" / "src" / "app.css").write_text("@import 'tailwindcss';", encoding="utf-8")
    ambient.mkdir()
    older = ambient / ".1111-00000000.node"
    older.write_bytes(b"left by an earlier run")
    given: list[Path] = []

    def fake_cli(command, **kwargs):
        scratch = Path(kwargs["env"]["TEMP"])
        assert kwargs["env"]["TMP"] == kwargs["env"]["TMPDIR"] == str(scratch) and scratch.is_dir()
        given.append(scratch)
        (scratch / ".2222-00000000.node").write_bytes(b"x" * 1024)  # what the real CLI leaves behind
        Path(command[command.index("-o") + 1]).write_text("/* css */", encoding="utf-8")

        class Done:
            returncode, stderr = 0, ""

        return Done()

    monkeypatch.setattr(build_css, "find_root", lambda: root)
    monkeypatch.setattr(build_css, "tailwind_binary", lambda: Path("tailwindcss"))
    monkeypatch.setattr(build_css.tempfile, "tempdir", str(ambient))  # pytest fixes this before the isolation applies: nothing may depend on it
    monkeypatch.setattr(build_css.subprocess, "run", fake_cli)

    assert build_css.ensure_css(force=True) is True
    assert given and not given[0].exists(), "the folder the compiler worked in is gone with what it unpacked"
    assert given[0].parent == root / ".cache" / "tmp", "and it was inside the project"
    assert older.exists(), "nothing else is touched"
