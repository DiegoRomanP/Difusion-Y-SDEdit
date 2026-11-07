from pc3.io import source_fingerprint


def test_code_fingerprint_ignores_documentation_but_detects_code_changes(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    original = source_fingerprint(tmp_path)
    (tmp_path / "README.md").write_text("Documentación, no código del experimento.")
    assert source_fingerprint(tmp_path) == original
    source.write_text("VALUE = 2\n")
    assert source_fingerprint(tmp_path) != original
