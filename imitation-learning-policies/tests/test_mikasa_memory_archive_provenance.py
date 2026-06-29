from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments" / "mikasa_method_archive" / "provenance"
REQUIRED_METHODS = {
    "no_memory",
    "top1_retrieval",
    "selector",
    "gru",
    "gru_burnin",
    "late_cue_direct_anchor",
}


def _read_tsv(path: Path):
    rows = path.read_text().strip().splitlines()
    header = rows[0].split("\t")
    return [dict(zip(header, row.split("\t"))) for row in rows[1:]]


def test_each_method_has_code_source():
    rows = _read_tsv(ARCHIVE / "method_code_sources.tsv")
    assert REQUIRED_METHODS <= {row["method"] for row in rows}
    for row in rows:
        assert row["source_code_path"].startswith(
            "/mnt/3fs1/data/tingwen.du/icra_method_dev/"
        )
        assert row["repo_target"].startswith("imitation-learning-policies/")


def test_each_core_method_has_result_source():
    rows = _read_tsv(ARCHIVE / "result_sources.tsv")
    methods = {row["method"] for row in rows}
    assert {
        "no_memory",
        "top1_retrieval",
        "selector",
        "gru",
        "late_cue_direct_anchor",
    } <= methods
    for row in rows:
        assert row["task"]
        assert row["score"]
        assert row["source_path"].startswith(
            "/mnt/3fs1/data/tingwen.du/icra_method_dev/"
        )


def test_checkpoint_sources_are_paths_not_large_files():
    rows = _read_tsv(ARCHIVE / "checkpoint_sources.tsv")
    assert {"top1_retrieval", "selector", "gru", "late_cue_direct_anchor"} <= {
        row["method"] for row in rows
    }
    for row in rows:
        assert row["checkpoint_path"].startswith(
            "/mnt/3fs1/data/tingwen.du/icra_method_dev/"
        )
        assert row["checkpoint_path"].endswith(".ckpt")
