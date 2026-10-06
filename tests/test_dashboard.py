"""P9.5: the dashboard skeleton's rules, checked.

- No app file contains the word "secrets" (the key directory, or Streamlit's
  ``st.secrets``), and no Streamlit secrets file exists.
- No measured number is typed into app code: string literals in app modules may
  contain digits only through `DIGIT_ALLOWLIST`, and numeric literals must be in
  `NUMBER_ALLOWLIST`. The scanner is shown to catch planted violations.
- The data layer reads only manifest-listed files under results/, figures/ and
  provenance/, re-hashes every read, and raises on a missing, changed or
  unlisted file.
- The committed manifest lists only git-tracked files, each matching on disk.
- Every page renders with the not-legal-evidence footer, and a data failure is
  shown as a visible error.
- The app imports only the standard library, Streamlit and itself, so
  app/requirements.txt covers it on Streamlit Community Cloud.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.data import ALLOWED_ROOTS, MANIFEST_PATH, MANIFEST_SCHEMA, DataIntegrityError, DataStore, check_path
from app.pages import PAGES
from src.utils.results import repo_root

APP = repo_root() / "app"

DIGIT_ALLOWLIST = (
    (r"\bP\d+(?:\.\d+)*\b", "task IDs such as P9.5, which name a task, not a measurement"),
    (r"SHA-256", "the hash function's name"),
    (r"/v\d+\b", "schema version suffixes such as zk-crown/dashboard-manifest/v1"),
    (r"\butf-8\b", "the text encoding name"),
    (r"\bsha256\b", "the manifest's field name for a SHA-256 digest"),
    (r"</?h[1-6]\b", "HTML heading tags"),
    (r"\bp\d+\.\d+_", "task prefixes of committed result file names, e.g. results/p4.9_master_table__"),
    (r"\bCIFAR-10\b", "the dataset's name"),
    (r"\b[Gg]roth16", "the proof system's name, also in record field names such as groth16_verified"),
    (r"\bBN254\b", "the elliptic curve's name"),
)
"""Digit patterns allowed inside app string literals. Everything else with a digit fails."""
FORMAT_SPEC = r"[+,]?\.\d+[fFeEg%]"
"""A string that is only a format specification (e.g. ``.2f``, ``+.2f``, ``.2%``) is allowed: it sets how a
computed number is shown, not a number."""
NUMBER_ALLOWLIST = {0, 1}
"""Numeric literals allowed in app code: indexing and 'plus one'. Literal arguments of ``time.sleep``
(animation pauses, never displayed) are also allowed; nothing else."""
NOT_SCANNED_FOR_NUMBERS = {"build_manifest.py"}
"""Local tool that writes the manifest (indent, exit codes); never imported by the running app (tested)."""


def app_python_files() -> list[Path]:
    return sorted(APP.rglob("*.py"))


def docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
        if isinstance(node, (ast.Module, ast.ClassDef)):
            # attribute docstrings: a bare string right after an assignment
            for prev, cur in zip(node.body, node.body[1:]):
                if isinstance(prev, (ast.Assign, ast.AnnAssign)) and isinstance(cur, ast.Expr) \
                        and isinstance(cur.value, ast.Constant) and isinstance(cur.value.value, str):
                    ids.add(id(cur.value))
    return ids


def sleep_arguments(tree: ast.AST) -> set[int]:
    """Numeric literals passed to ``time.sleep``: animation timing, never displayed."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "sleep")
                or (isinstance(node.func, ast.Name) and node.func.id == "sleep")):
            ids |= {id(a) for a in node.args if isinstance(a, ast.Constant)}
    return ids


def literal_violations(source: str, name: str, scan_numbers: bool = True) -> list[str]:
    tree = ast.parse(source)
    skip = docstring_nodes(tree) | sleep_arguments(tree)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        value = node.value
        if isinstance(value, str):
            if re.fullmatch(FORMAT_SPEC, value):
                continue
            stripped = value
            for pattern, _ in DIGIT_ALLOWLIST:
                stripped = re.sub(pattern, "", stripped)
            if re.search(r"\d", stripped):
                problems.append(f"{name}:{node.lineno}: digit in string literal {value[:60]!r}")
        elif scan_numbers and isinstance(value, (int, float, complex)) and not isinstance(value, bool):
            if value not in NUMBER_ALLOWLIST:
                problems.append(f"{name}:{node.lineno}: numeric literal {value!r}")
    return problems


# --- secrets ------------------------------------------------------------------------


def test_no_app_file_mentions_the_secrets_directory():
    roots = [APP, repo_root() / ".streamlit"]
    hits = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                text = path.read_text(encoding="utf-8", errors="replace")
                if re.search(r"secrets", text, re.IGNORECASE):
                    hits.append(str(path.relative_to(repo_root())))
    assert hits == [], f"app files refer to secrets: {hits}"
    assert not (repo_root() / ".streamlit" / "secrets.toml").exists()


def test_secrets_scan_catches_a_planted_reference(tmp_path):
    planted = 'KEY = open("../secrets/K.bin", "rb").read()\n'
    assert re.search(r"secrets", planted, re.IGNORECASE)
    assert re.search(r"secrets", "token = st.secrets['k']", re.IGNORECASE)


def test_data_layer_refuses_secret_and_escaping_paths(tmp_path):
    store = _store(tmp_path, {"results/a.json": b"{}"})
    for bad in ("secrets/K.bin", "results/../secrets/K.bin", "../secrets/K.bin", "/etc/passwd",
                "C:/Windows/win.ini", "results\\a.json", "data/cifar-10-batches-py/test_batch", "./results/a.json",
                "app/manifest.json", ""):
        with pytest.raises(DataIntegrityError):
            store.read_bytes(bad)


# --- numeric literals ---------------------------------------------------------------


def test_no_measured_numbers_typed_into_app_code():
    problems = []
    for path in app_python_files():
        rel = str(path.relative_to(repo_root()))
        problems += literal_violations(path.read_text(encoding="utf-8"), rel,
                                       scan_numbers=path.name not in NOT_SCANNED_FOR_NUMBERS)
    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize("source", [
    'st.metric("Test accuracy", "90.85%")\n',
    'st.write(f"WDR {fired} of 100")\n',
    'st.caption("z = 10.29")\n',
    'st.metric("Accuracy", 0.9085)\n',
    'cols = st.columns(4)\n',
    'TEXT = "Bitcoin block 969627"\n',
    'st.html("<h2>Fired: 100</h2>")\n',
    'st.write("sha256 of 7 files")\n',
    'st.write(f"accuracy {acc:.2f} on 10000 images")\n',
    'LABEL = "90.85%"\n',
    'NAME = "results/p4.9_master_table__seed1337__20261002T171216+0000.json"\n',
    'st.write("CIFAR-10 test accuracy 94.37")\n',
    'st.write("Groth16 proof of 806 bytes")\n',
    'time.sleep(0.5)\nst.progress(0.5)\n',
])
def test_scanner_catches_planted_numbers(source):
    assert literal_violations(source, "planted.py")


@pytest.mark.parametrize("source", [
    '"""Docstring with P9.5 and 91.20% is documentation, not displayed."""\nx = "fine"\n',
    'chip = "Planned · P9.13"\n',
    'note = "checked against their recorded SHA-256"\n',
    'SCHEMA = "zk-crown/dashboard-manifest/v1"\n',
    'last = items[-1]\nfirst = items[0]\n',
    'st.write(f"accuracy {acc:.2%}, drop {d:+.2f} pp")\n',
    'PREFIX = "results/p4.9_master_table__"\n',
    'st.write("third-party CIFAR-10 models")\n',
    'time.sleep(0.6)\nflag = stat["groth16_verified"]\n',
])
def test_scanner_allows_the_documented_exceptions(source):
    assert literal_violations(source, "ok.py") == []


def test_build_manifest_is_not_imported_by_the_app():
    for path in app_python_files():
        if path.name != "build_manifest.py":
            assert "build_manifest" not in path.read_text(encoding="utf-8"), path


# --- data layer ---------------------------------------------------------------------


def _store(root: Path, files: dict[str, bytes]) -> DataStore:
    entries = {}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "files": entries}), encoding="utf-8")
    return DataStore(repo_root=root, manifest_path=manifest)


def test_verified_read(tmp_path):
    store = _store(tmp_path, {"results/a.json": b'{"x": 1}', "figures/f.png": b"\x89PNG", "provenance/p.json": b"[]"})
    assert store.read_json("results/a.json") == {"x": 1}
    assert store.read_bytes("figures/f.png") == b"\x89PNG"
    assert all(c.ok for c in store.verify_all())


def test_changed_file_is_refused_every_time_it_is_read(tmp_path):
    store = _store(tmp_path, {"results/a.json": b'{"x": 1}'})
    assert store.read_json("results/a.json") == {"x": 1}
    (tmp_path / "results" / "a.json").write_bytes(b'{"x": 2}')
    with pytest.raises(DataIntegrityError, match="does not match the recorded"):
        store.read_json("results/a.json")
    (tmp_path / "results" / "a.json").write_bytes(b'{"x": 1}\r\n')
    with pytest.raises(DataIntegrityError, match="does not match the recorded"):
        store.read_bytes("results/a.json")
    check = store.verify("results/a.json")
    assert not check.ok and check.sha256 is None


def test_missing_and_unlisted_files(tmp_path):
    store = _store(tmp_path, {"results/a.json": b"{}"})
    (tmp_path / "results" / "b.json").write_bytes(b"{}")
    with pytest.raises(DataIntegrityError, match="not listed"):
        store.read_bytes("results/b.json")
    (tmp_path / "results" / "a.json").unlink()
    with pytest.raises(DataIntegrityError, match="missing"):
        store.read_bytes("results/a.json")


def test_malformed_manifests_are_refused(tmp_path):
    good = {"sha256": "0" * 64, "bytes": 1}
    for manifest in ({"schema": "other", "files": {"results/a": good}},
                     {"schema": MANIFEST_SCHEMA, "files": {}},
                     {"schema": MANIFEST_SCHEMA, "files": {"secrets/K.bin": good}},
                     {"schema": MANIFEST_SCHEMA, "files": {"results/../x": good}},
                     {"schema": MANIFEST_SCHEMA, "files": {"results/a": {"sha256": "abc", "bytes": 1}}},
                     {"schema": MANIFEST_SCHEMA, "files": {"results/a": {"sha256": "A" * 64, "bytes": 1}}}):
        path = tmp_path / "m.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(DataIntegrityError):
            DataStore(repo_root=tmp_path, manifest_path=path)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        DataStore(repo_root=tmp_path, manifest_path=tmp_path / "absent.json")


def test_check_path_canonical_form():
    assert check_path("results/p9.4.json") == "results/p9.4.json"
    assert check_path("figures/a/b.png") == "figures/a/b.png"
    assert set(ALLOWED_ROOTS) == {"results", "figures", "provenance"}


def test_committed_manifest_lists_only_tracked_files_that_match():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema"] == MANIFEST_SCHEMA
    tracked = set(subprocess.run(["git", "ls-files", "--", *ALLOWED_ROOTS], cwd=repo_root(), check=True,
                                 capture_output=True, text=True).stdout.splitlines())
    assert set(manifest["files"]) <= tracked
    assert not any(name.startswith("secrets") for name in manifest["files"])
    bad = [c.path for c in DataStore().verify_all() if not c.ok]
    assert bad == [], f"files differ from the manifest: {bad}"


# --- pages ---------------------------------------------------------------------------


def _app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)


def _html(at) -> list[str]:
    return [h.proto.body for h in at.get("html")]


def test_not_legal_text_matches_the_auditor():
    from app.ui import NOT_LEGAL
    from src.auditor.grading import NOT_LEGAL as AUDITOR_NOT_LEGAL

    assert NOT_LEGAL == AUDITOR_NOT_LEGAL


def test_every_page_renders_with_the_footer():
    from app.ui import NOT_LEGAL

    at = _app_test().run()
    assert not at.exception
    for spec in PAGES:
        at.switch_page(f"views/{spec.slug}.py").run()
        assert not at.exception, spec.slug
        bodies = _html(at)
        assert any(NOT_LEGAL in b and "zk-footer" in b for b in bodies), spec.slug
        title = "zk-hero-title" if spec.slug == "overview" else f">{spec.title}</h1>"
        assert any(title in b for b in bodies), spec.slug
        assert not at.error, f"{spec.slug}: {[e.value for e in at.error]}"


def test_integrity_page_shows_a_changed_file_as_a_visible_error(monkeypatch, tmp_path):
    import app.data as data
    import app.ui as ui

    _store(tmp_path, {"results/a.json": b"{}", "figures/f.png": b"png"})
    (tmp_path / "results" / "a.json").write_bytes(b'{"tampered": true}')
    monkeypatch.setattr(ui, "_cached_store", lambda: DataStore(repo_root=tmp_path,
                                                               manifest_path=tmp_path / "manifest.json"))
    at = _app_test().run()
    at.switch_page("views/integrity.py").run()
    errors = [e.value for e in at.error]
    assert any("results/a.json" in e and "does not match the recorded" in e for e in errors), errors
    assert not at.success
    assert data.MANIFEST_PATH == MANIFEST_PATH


def test_page_files_match_the_registry():
    files = {p.stem for p in (APP / "views").glob("*.py") if p.stem != "__init__"}
    assert files == {p.slug for p in PAGES}
    assert sum(p.default for p in PAGES) == 1


# --- hosting -------------------------------------------------------------------------


def test_app_imports_only_stdlib_streamlit_and_itself():
    allowed_third_party = {"streamlit"}
    stdlib = set(sys.stdlib_module_names)
    found = set()
    for path in app_python_files():
        if path.name == "build_manifest.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.add(node.module.split(".")[0])
    outside = {m for m in found if m not in stdlib and m not in allowed_third_party and m != "app"}
    assert outside == set(), outside


def test_app_requirements_are_pinned_and_minimal():
    lines = [ln.strip() for ln in (APP / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert lines and all("==" in ln for ln in lines)
    assert {ln.split("==")[0] for ln in lines} == {"streamlit"}
    import streamlit

    assert f"streamlit=={streamlit.__version__}" in lines
