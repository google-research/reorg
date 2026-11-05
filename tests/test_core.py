# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests: no API calls, no benchmark data.

    python3 -m pytest -q tests/        or        python3 tests/test_core.py
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reorg import prompts  # noqa: E402
from reorg.domain import load_domain  # noqa: E402
from reorg.files import files_digest, read_ids, write_jsonl  # noqa: E402

DATA = Path(tempfile.gettempdir()) / "reorg-tests-data"          # never read: no test touches benchmark data


def _domains():
    return {name: load_domain(name, DATA) for name in ("gdpval", "jobbench", "lab")}


# ---- prompts ---------------------------------------------------------------

def test_prompts_match_manifest():
    """Every template matches its sha256 in reorg/prompts/manifest.json."""
    listed = prompts.manifest()
    for name, entry in listed.items():
        assert hashlib.sha256((prompts.PROMPTS / name).read_bytes()).hexdigest() == entry["sha256"], name
    on_disk = {path.relative_to(prompts.PROMPTS).as_posix() for path in prompts.PROMPTS.rglob("*")
               if path.suffix in (".md", ".json")}
    assert on_disk - set(listed) == {"README.md", "manifest.json"}
    assert len(listed) == 31                                     # 26 templates + 5 output schemas


def test_prompt_boxes_and_roles():
    boxes = {entry["paper"] for entry in prompts.manifest().values()}
    assert boxes == {f"P{i}" for i in range(1, 12)} | {f"R{i}" for i in range(1, 11)}
    assert prompts.role_ids() == sorted([
        "same_job_teammate", "direct_manager", "upstream_requester", "cross_functional_partner",
        "technical_customer", "implementation_partner", "procurement_evaluator",
        "legal_risk_reviewer", "internal_auditor", "experienced_mentor"])




# ---- task sets -------------------------------------------------------------

def test_read_ids_keeps_file_order():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ids.txt"
        path.write_text("# comment\nz\n\na\nm\n")
        assert read_ids(path) == ["z", "a", "m"]


def test_task_set_sizes_and_disjointness():
    d = _domains()
    g, j, lab = d["gdpval"], d["jobbench"], d["lab"]
    assert (len(g.eval_ids()), len(g.adapt_ids()), len(g.heldout_ids())) == (185, 20, 165)
    assert (len(j.eval_ids()), len(j.adapt_ids()), len(j.heldout_ids())) == (128, 20, 108)
    assert (len(lab.eval_ids()), len(lab.adapt_ids()), len(lab.heldout_ids())) == (200, 20, 200)
    for dom in (g, j):
        adapt, heldout = set(dom.adapt_ids()), set(dom.heldout_ids())
        assert not adapt & heldout and adapt | heldout == set(dom.eval_ids())
    assert not set(lab.adapt_ids()) & set(lab.eval_ids())
    for dom in d.values():
        assert len(dom.smoke_ids()) == 2 and set(dom.smoke_ids()) <= set(dom.adapt_ids())


def test_jobbench_ids_are_split_qualified():
    j = _domains()["jobbench"]
    ids = j.eval_ids()
    assert all(task_id.split("/", 1)[0] in ("main", "easy") for task_id in ids)
    assert sum(task_id.startswith("main/") for task_id in ids) == 65
    adapt = j.adapt_ids()
    assert sum(task_id.startswith("main/") for task_id in adapt) == 10


def test_domains_load_side_by_side():
    d = _domains()
    modules = {name: sys.modules[f"data.{name}.prepare_data"] for name in d}
    assert len({id(module) for module in modules.values()}) == 3
    for name, dom in d.items():
        assert dom.name == name and dom.data == DATA.resolve() / name


def test_each_load_is_a_fresh_domain():
    first, second = load_domain("gdpval", Path("/a")), load_domain("gdpval", Path("/b"))
    assert first is not second and first.data == Path("/a/gdpval") and second.data == Path("/b/gdpval")


# ---- data preparation logic --------------------------------------------------

def test_gdpval_cohort_is_tasks_with_expert_deliverables():
    _domains()
    cohort = sys.modules["data.gdpval.prepare_data"].cohort
    rows = [{"task_id": "b", "deliverable_files": ["x.xlsx"]},
            {"task_id": "a", "deliverable_files": []},
            {"task_id": "c", "deliverable_files": ["y.docx", "z.pdf"]}]
    assert cohort(rows) == ["b", "c"]


def _lab_rows():
    """Three practice areas of 12 families; every fourth family has 3 scenarios."""
    rows = []
    for area in ("corporate", "litigation", "tax"):
        for fam in range(12):
            family = f"{area}/family-{fam}"
            ids = [f"{family}/scenario-{v}" for v in (1, 2, 3)] if fam % 4 == 0 else [family]
            rows += [{"task_id": task_id, "family_id": family, "stratum": f"{area}::memo::.docx"} for task_id in ids]
    return rows


def test_lab_split_is_seeded_and_keeps_families_together():
    _domains()
    assign = sys.modules["data.lab.prepare_data"].assign_splits
    rows = _lab_rows()
    first = assign(rows, test_size=10, valid_size=10, seed=20260717)
    assert first == assign(rows, test_size=10, valid_size=10, seed=20260717)
    split_of_family = {}
    for row in first:
        assert split_of_family.setdefault(row["family_id"], row["split"]) == row["split"]
    sizes = {s: sum(row["split"] == s for row in first) for s in ("train", "valid", "test")}
    assert abs(sizes["test"] - 10) <= 2 and abs(sizes["valid"] - 10) <= 2
    assert [row["task_id"] for row in first] == [row["task_id"] for row in rows]


def test_lab_deliverables_parsed_from_instructions():
    _domains()
    lab = sys.modules["data.lab.prepare_data"]
    text = "Draft the memo.\n\n### Output:\n1. `client_memo.docx`\n2. `issues_list.xlsx`\n\nThanks."
    assert lab._deliverables({"instructions": text}) == (
        ["client_memo.docx", "issues_list.xlsx"], "instructions")
    assert lab._deliverables({"instructions": "Write it.", "deliverables": {"memo": "memo.docx"}}) == (
        ["memo.docx"], "task_json")
    assert lab._deliverables({"instructions": "Write a memo."}) == (
        ["output.docx"], "fallback_output_docx")


# ---- files -----------------------------------------------------------------

def test_files_digest_is_sha256sum_format():
    entries = {"b.txt": "2" * 64, "a.txt": "1" * 64}
    expected = hashlib.sha256(f"{'1' * 64}  a.txt\n{'2' * 64}  b.txt\n".encode()).hexdigest()
    assert files_digest(entries) == expected



def test_jsonl_keeps_unicode_unescaped():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        write_jsonl(path, [{"title": "Société générale", "n": 1}])
        assert path.read_text(encoding="utf-8") == json.dumps({"title": "Société générale", "n": 1},
                                                              ensure_ascii=False) + "\n"


def test_manifests_pin_sources():
    d = _domains()
    for name, dom in d.items():
        manifest = json.loads((dom.root / "splits" / "manifest.json").read_text())
        assert len(manifest.get("revision", "")) == 40, name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
