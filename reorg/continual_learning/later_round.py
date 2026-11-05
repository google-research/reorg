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
"""Learning later-round retrieval (App. A.3, "Later Rounds and Deployment").

With the one-round library L and the lessons in M frozen, each adaptation task
x is revisited:

    s_1       first-round retrieval (P5), its single top skill
    A_1, H_1  A_0 = W(x), then one round with s_1                   the parent state
    z         the work-state projection (P8) of (x, H_1)
    S         8 unused nonempty skills of L, sampled without replacement: slots go
              round-robin over the populated node-count levels in a seeded order,
              each level drawn in its own seeded order
    A_2^s     one round with s from (A_1, H_1), for every s in S      a branch
    ranking   the strict branch artifacts and A_1 itself as STOP, in groups of
              four (P6) in which every branch meets STOP; branches whose judged
              files are identical, to each other or to A_1, are judged once
    lesson    P11 over the comparisons, given x and z

A ranking needs STOP and four other distinct artifacts. The continuation
fitness of z is the share of distinct branch artifacts ranked above STOP, ties
counted half. The precedents in M (precedents.md) list every task's z, its
fitness and its lesson in the document format the later-round planner reads;
L and lessons.md are unchanged. In model-facing text, ranking groups are named
gate3_<task alias>_group_### and branches "Communication approach k" in
sampling order.

    <out>/library/                      a copy of (L, M) with precedents.md
    <out>/tasks/<task_id>/              retrieval.json, parent/, state.json,
                                        branches/<fingerprint[:16]>/, ranking/,
                                        lesson/, record.json

record.json marks a finished task; a rerun reuses every finished step.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .. import llm, planner, prompts
from ..domain import Domain, TaskPacket
from ..files import files_digest, read_json, sha256_file, write_json
from ..library import Library, task_alias as alias
from ..reflection_skill.chronicle import WorkChronicle
from ..reflection_skill.graph import ReflectionSkill
from ..reflection_skill.reflect import Runtime, reflect
from . import ranking, search

STOP = "STOP"
# Random-stream namespaces; they fix the sampled skills and the groups.
SAMPLING_POLICY = "colleague_vertex_count_seeded_max_min.v0.1"
SCHEDULER = "later_round_stop_anchored.v0.1"
STATE_KEYS = ("task_summary", "what_was_suggested", "what_was_done")

SYSTEM = "Project a task-local communication lesson and return only valid JSON."
EVIDENCE_LIMITS = ("This is task-local ordinal evidence from one resulting version per communication "
                   "approach. Only completed rankings and judge explanations are supplied here, not "
                   "the raw later-round artifacts. Judge explanations may disagree; preserve any "
                   "conflict instead of resolving it. Do not make a causal, absolute-quality, or "
                   "held-out generalization.")
# A lesson must not name aliases, judge labels, controller IDs, digests or paths.
INTERNAL = re.compile(r"(?:\bflow_[0-9]{3}\b|\bartifact_[A-Z]\b|\b(?:candidate|group|slot)[_-][A-Za-z0-9_-]+|"
                      r"\b[a-f0-9]{64}\b|/(?:home|tmp|var/tmp)/)", re.IGNORECASE)

GUIDANCE = """- Base the recommendation on the current work, including what is already done
  and what still needs attention.
- Leaving the work unchanged is a real option.
- If another review would help, prefer a communication flow that directly
  matches the remaining needs.
- Do not repeat work that earlier reviewers and the owner have already
  completed.
- More reviewers, more messages, or a more complicated flow are not
  automatically better.
- Treat each example below as an observation from one work state, not a rule
  that always applies."""
PRECEDENT_LIMITS = ("One task-local W1 state, one frozen candidate pool, and ordinal "
                    "artifact-only comparisons; no causal or held-out generalization claim.")

# The later-round judge reads the request itself; JobBench's also carries this note.
NETWORKLESS_NOTE = """Some tasks allowed the solver to use web retrieval, but this ranking session has no
network access and no search transcript. Judge each finished submission as a
self-contained professional deliverable. Do not invent or independently assume
external facts. For claims that cannot be checked from the visible task,
references, and submission, compare the quality of visible support, sourcing,
internal consistency, caveats, and task fit rather than pretending to verify
the underlying fact."""


# ---- task identity and sampling ----------------------------------------------

def task_seed(task_id: str, seed: int) -> int:
    """The per-task sampling seed, derived from the experiment seed and the task alias."""
    payload = {"schema": "later_round_task_candidate_seed.v0.1", "experiment_seed": seed, "task_fs_id": alias(task_id)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16)


def sample(skills: list[ReflectionSkill], used: str, count: int, seed: int) -> list[ReflectionSkill]:
    """Up to `count` nonempty skills other than the used one, without replacement, round-robin over
    node-count levels. Only the used skill ID is excluded (L holds one skill per fingerprint)."""
    levels: dict[int, list[ReflectionSkill]] = defaultdict(list)
    for skill in sorted(skills, key=lambda skill: skill.skill_id):
        if skill.nodes and skill.skill_id != used:
            levels[len(skill.nodes)].append(skill)
    order = sorted(levels)
    random.Random(f"{SAMPLING_POLICY}:{seed}:level-order").shuffle(order)
    for level in order:
        random.Random(f"{SAMPLING_POLICY}:{seed}:level:{level}").shuffle(levels[level])
    rounds = itertools.zip_longest(*(levels[level] for level in order))
    return [skill for row in rounds for skill in row if skill is not None][:count]


# ---- execution ---------------------------------------------------------------

def branch(skill: ReflectionSkill, runtime: Runtime, submit: Callable[[Path, Path], list[str]], parent: Path,
           out_dir: Path, parallelism: int, retries: int) -> str:
    """Round 2 with `skill` from the parent's A_1 and H_1, and its submission (out_dir/submission).
    Returns the status; only "succeeded" is ranked. Kept in out_dir/result.json."""
    if (out_dir / "result.json").exists():
        return read_json(out_dir / "result.json")["status"]
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for prior in ("a0", "round_1"):                 # the parent's rounds, where runtimes find earlier evidence
        (out_dir / prior).symlink_to((parent / prior).resolve(), target_is_directory=True)
    chronicle = WorkChronicle(out_dir / "work_chronicle.md", (parent / "work_chronicle.md").read_text(encoding="utf-8"))
    # search.execute_skill keeps the parent's revised artifact A_1 in round_1/artifact.
    result = reflect(skill, runtime, parent / "round_1" / "artifact", chronicle, out_dir / "round_2", parallelism,
                     retries)
    write_json(out_dir / "round_2" / "round.json", result.to_dict())
    if result.status == "strict":
        status = "succeeded" if submit(result.artifact, out_dir / "submission") else "invalid_output"
    elif result.records[-1]["status"] == "infrastructure_error":
        status = "infrastructure_failed"
    else:
        status = "model_quality_failed"
    write_json(out_dir / "result.json", {"skill": skill.to_dict(), "status": status})
    return status


def collapse(artifacts: dict[str, Path]) -> dict[str, str]:
    """candidate -> representative. Candidates with identical judged files share one representative:
    STOP if it is among them, else the smallest candidate ID."""
    by_content: dict[str, list[str]] = defaultdict(list)
    for name, root in artifacts.items():
        files = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}
        by_content[files_digest(files)].append(name)
    return {name: STOP if STOP in names else min(names) for names in by_content.values() for name in names}


# ---- ranking -----------------------------------------------------------------

def stop_groups(candidates: list[str], seed: int) -> list[tuple[str, ...]]:
    """Groups of four over STOP and 4 to 11 actions, in which every action meets STOP. Four actions:
    STOP with each three of them. Otherwise STOP with each three-chunk of the seeded action order
    (the last padded), more STOP groups until STOP appears three times, then rotated four-chunks of
    actions until every candidate appears three times and the design is robust."""
    actions = sorted(candidate for candidate in candidates if candidate != STOP)
    random.Random(seed).shuffle(actions)
    position = {action: index for index, action in enumerate(actions)}
    groups: list[tuple[str, ...]] = []
    count: Counter = Counter()

    def add(group: tuple[str, ...]) -> bool:
        if any(sorted(group) == sorted(other) for other in groups):
            return False
        groups.append(group)
        count.update(group)
        return True

    def deficit(group: tuple[str, ...]) -> int:
        return sum(max(0, ranking.MIN_APPEARANCES - count[candidate]) for candidate in group)

    if len(actions) == 4:
        for omitted in actions:
            add((STOP, *(action for action in actions if action != omitted)))
        return groups
    for start in range(0, len(actions), 3):
        chunk = actions[start:start + 3]
        chunk += sorted((a for a in actions if a not in chunk), key=lambda a: (count[a], position[a]))[:3 - len(chunk)]
        add((STOP, *chunk))
    while count[STOP] < ranking.MIN_APPEARANCES:
        for trio in sorted(itertools.combinations(actions, 3),
                           key=lambda g: (-deficit(g), sum(count[c] for c in g), tuple(position[c] for c in g))):
            if add((STOP, *trio)):
                break
    support: list[tuple[str, ...]] = []
    for offset in range(len(actions)):
        rotated = actions[offset:] + actions[:offset]
        for start in range(0, len(rotated), ranking.GROUP_SIZE):
            chunk = rotated[start:start + ranking.GROUP_SIZE]
            chunk = (chunk + [a for a in rotated if a not in chunk])[:ranking.GROUP_SIZE]
            if all(sorted(chunk) != sorted(other) for other in groups + support):
                support.append(tuple(chunk))
    everyone = [STOP, *actions]
    while min(count[c] for c in everyone) < ranking.MIN_APPEARANCES or not ranking.robust(everyone, groups):
        # max keeps the first best group in support order.
        add(max((group for group in support if group not in groups),
                key=lambda g: (deficit(g), -sum(count[c] for c in g), tuple(-position[c] for c in g))))
    return groups


def judge_material(domain_name: str, packet: TaskPacket) -> ranking.Material:
    """The task as the later-round judge sees it."""
    names = [Path(name).name for name in packet.deliverables]
    text = (f"{packet.request.rstrip()}\n\n# Ranking constraint\n\n{NETWORKLESS_NOTE}\n" if domain_name == "jobbench"
            else packet.request)
    return ranking.Material(text, {"expected_count": len(names) or None, "expected_basenames": names,
                                   "expected_suffixes": [Path(name).suffix.lower() for name in names],
                                   "visibility_policy": "artifact_only_networkless"}, packet.references)


def stop_relations(record: dict) -> Counter:
    """better / tied / worse than STOP over the ranked branch artifacts."""
    tier = {name: index for index, members in enumerate(record["tiers"]) for name in members}
    return Counter("better" if tier[name] < tier[STOP] else "worse" if tier[name] > tier[STOP] else "tied"
                   for name in tier if name != STOP)


def continuation(relations: Counter) -> float:
    return (relations["better"] + 0.5 * relations["tied"]) / sum(relations.values())


# ---- the lesson (P11) --------------------------------------------------------

def deblind(text: str, labels: dict[str, str], titles: dict[str, str]) -> str:
    """A judge's explanation with its artifact labels replaced by approach titles."""
    by_letter = {label.removeprefix("artifact_").upper(): titles[name] for name, label in labels.items()}

    def listed(match: re.Match) -> str:
        letters = [part.strip().upper() for part in re.sub(r"\band\b", ",", match.group(1), flags=re.IGNORECASE)
                   .split(",") if part.strip()]
        if not all(letter in by_letter for letter in letters):
            return match.group(0)
        return ", ".join(by_letter[letter] for letter in letters)

    text = re.sub(r"\b(?i:Artifacts?)\s+([A-Z](?:\s*,\s*[A-Z])*(?:\s*,?\s*and\s+[A-Z])?)", listed, text.strip())
    for letter, title in by_letter.items():
        text = re.sub(rf"\bartifact_{letter}\b", title, text, flags=re.IGNORECASE)
        text = re.sub(rf"\b{letter}\b", title, text)
    return text


def lesson_prompt(task: str, state: dict, sampled: list[ReflectionSkill], statuses: dict[str, str],
                  representative: dict[str, str], record: dict) -> str:
    """P11 for one task: the branches, the STOP ranking with every judge's explanation, and the
    calculated summary. `record` is the ranking of the representatives; `statuses` the branches'."""
    titles = {STOP: "Leaving the work unchanged",
              **{skill.skill_id: f"Communication approach {k}" for k, skill in enumerate(sampled, 1)}}

    def order(tiers: list[list[str]]) -> str:
        return " > ".join(" = ".join(titles[name] for name in tier) for tier in tiers)

    flows = [f"### {titles[skill.skill_id]}\n- On the current work, {planner.communication_flow(skill)}. "
             "After this communication, the Task Owner performs one normal revision." for skill in sampled]
    flows.append("### Leaving the work unchanged\n"
                 "- No additional reviewer, discussion, or Task Owner revision is performed.")
    expanded = [[name for name in sorted(representative) if representative[name] in tier] for tier in record["tiers"]]
    comparisons = ["Aggregate order across the completed comparisons:", order(expanded)]
    for index, group in enumerate(record["groups"], start=1):
        explanation = deblind(group["decisive_difference"], group["labels"], titles)
        comparisons += ["", f"Judge comparison {index}:", order(group["tiers"]),
                        f"Judge's decisive explanation: {explanation}"]
        if group["hard_invalid"]:
            comparisons.append(f"Judge-marked invalid: {', '.join(titles[name] for name in group['hard_invalid'])}.")
    relations = stop_relations(record)
    summary = (f"Among {sum(relations.values())} artifact-distinct steering results, {relations['better']} ranked "
               f"above leaving the work unchanged, {relations['tied']} tied it, and {relations['worse']} ranked below "
               f"it. The programmatic continuation measure is {continuation(relations)}. This is the fraction of "
               "artifact-distinct steering results above leaving unchanged, with half credit for a tie; it is not an "
               "absolute quality score for the unchanged work.")
    excluded = sorted((name, status) for name, status in statuses.items() if status != "succeeded")
    if excluded:
        summary += " Excluded from quality ranking: " + "; ".join(f"{titles[n]} ({s})" for n, s in excluded) + "."
    return planner.render("learning/later_round_lesson.md", {
        "TASK": task, "CURRENT_WORK_SUMMARY": json.dumps(state, indent=2, ensure_ascii=False),
        "COMMUNICATION_FLOWS": "\n\n".join(flows), "COMPARISON_RESULTS": "\n".join(comparisons),
        "CALCULATED_SUMMARY": summary, "EVIDENCE_LIMITS": EVIDENCE_LIMITS})


def check_lesson(reply: dict) -> None:
    if any(INTERNAL.search(value) for value in reply.values()):
        raise ValueError("the lesson exposes an internal identifier")


# ---- precedents --------------------------------------------------------------

def precedents(entries: list[dict]) -> str:
    """precedents.md: the guidance and one example per work state, ordered by task alias."""
    blocks = ["# Guidance and examples from earlier work states", "", "## How to choose another communication flow",
              "", GUIDANCE, "", "## Examples from observed work states"]
    for number, entry in enumerate(sorted(entries, key=lambda entry: alias(entry["task_id"])), start=1):
        state, lesson = entry["state"], entry["lesson"]
        blocks += ["", f"### state {number}", "",
                   f"- Current work state: {state['task_summary']} Suggested: {' '.join(state['what_was_suggested'])} "
                   f"Done: {' '.join(state['what_was_done'])}",
                   f"- Leaving the work unchanged: continuation fitness `{entry['continue_fitness']}` "
                   "in this frozen pool.",
                   f"- Helpful communication flows if steering: {lesson['helpful_communication_flows_if_steering']}",
                   "- Helpful communication-flow features if steering: "
                   f"{lesson['helpful_communication_flow_features_if_steering']}",
                   f"- Evidence limits: {PRECEDENT_LIMITS}"]
    return "\n".join(blocks).rstrip() + "\n"


# ---- one task ----------------------------------------------------------------

def _cached(path: Path, compute: Callable[[], Any]) -> Any:
    """compute() once; the JSON result is kept at `path` and reused on a rerun."""
    if not path.exists():
        write_json(path, compute())
    return read_json(path)


def learn_task(domain: Domain, task_id: str, *, library: Library, task_dir: Path, hooks: Any, profile: llm.ModelProfile,
               config: dict) -> dict:
    """The parent state, the branches, their STOP ranking and the lesson of one adaptation task."""
    settings = config["later_round"]
    packet = domain.load_task(task_id)
    temperature = settings["temperature"].get(profile.provider)
    top, reason = _cached(task_dir / "retrieval.json", lambda: planner.select_first_round(
        profile, packet.request, library, top_k=settings["retrieve_top_k"], temperature=temperature,
        log_path=task_dir / "calls.jsonl")[0])
    record = {"task_id": task_id, "retrieved": [top, reason]}
    runtime = hooks.make_runtime(domain, task_id, profile, config)

    def submit(artifact: Path, out: Path) -> list[str]:
        return hooks.submission(domain, task_id, artifact, out)

    parent = task_dir / "parent"
    stop = search.execute_skill(library.skill(top), runtime, submit, parent, config["stakeholder_parallelism"],
                                config["retries"])
    if stop is None:
        return {**record, "status": "parent_ineligible"}

    def project() -> dict:
        state = planner.project_work_state(profile, packet.request, (parent / "work_chronicle.md").read_text(
            encoding="utf-8"), temperature=temperature, log_path=task_dir / "calls.jsonl")
        return {key: state[key] for key in STATE_KEYS}          # the key order P11 shows

    state = _cached(task_dir / "state.json", project)
    sampled = sample(library.skills(), top, settings["sampled_skills"], task_seed(task_id, settings["sampling_seed"]))
    out = {skill.skill_id: task_dir / "branches" / skill.fingerprint()[:16] for skill in sampled}
    with ThreadPoolExecutor(max_workers=max(1, min(config["candidate_parallelism"], len(sampled)))) as pool:
        statuses = dict(zip(out, pool.map(lambda skill: branch(
            skill, runtime, submit, parent, out[skill.skill_id], config["stakeholder_parallelism"], config["retries"]),
            sampled)))
    artifacts = {STOP: stop, **{name: out[name] / "submission" for name, status in statuses.items()
                                if status == "succeeded"}}
    representative = collapse(artifacts)
    record.update(statuses=statuses, representative=representative)
    distinct = sorted(set(representative.values()))
    if len(distinct) < settings["min_distinct_artifacts"]:
        return {**record, "status": "too_few_artifacts"}

    # Presentation draws are namespaced by the scheduler ID.
    groups = ranking.present(f"gate3_{alias(task_id)}", stop_groups(distinct, settings["grouping_seed"]),
                             f"{SCHEDULER}:{settings['grouping_seed']}")
    judge = ranking.Judge(replace(profile, temperature=profile.temperature if temperature is None else temperature),
                          **config["judge"])
    ranked = ranking.rank(groups, {name: artifacts[name] for name in distinct},
                          task=judge_material(domain.name, packet), judge=judge, out_dir=task_dir / "ranking")
    prompt = lesson_prompt(packet.request, state, sampled, statuses, representative, ranked)
    (task_dir / "lesson").mkdir(parents=True, exist_ok=True)
    (task_dir / "lesson" / "prompt.md").write_text(prompt, encoding="utf-8", newline="\n")
    schema = json.loads(prompts.load("learning/later_round_lesson.schema.json"))
    lesson = _cached(task_dir / "lesson" / "reply.json", lambda: llm.complete_json(
        profile, system=SYSTEM, prompt=prompt, schema=schema, temperature=temperature,
        log_path=task_dir / "lesson" / "calls.jsonl", validate=check_lesson))
    return {**record, "status": "precedent", "tiers": ranked["tiers"],
            "precedent": {"task_id": task_id, "state": state, "continue_fitness": continuation(stop_relations(ranked)),
                          "lesson": lesson}}


def run(domain: Domain, task_ids: list[str], *, library: Library, out_dir: Path, hooks: Any, profile: llm.ModelProfile,
        config: dict) -> list[dict]:
    """Learn the precedents of the frozen `library` over the adaptation stream; out_dir/library is
    the library with its precedents.md. Tasks run concurrently; a failed task fails the run once the
    others finish, and a rerun resumes it."""
    def one(task_id: str) -> dict:
        task_dir = out_dir / "tasks" / task_id
        if not (task_dir / "record.json").exists():
            write_json(task_dir / "record.json", learn_task(domain, task_id, library=library, task_dir=task_dir,
                                                            hooks=hooks, profile=profile, config=config))
        return read_json(task_dir / "record.json")

    with ThreadPoolExecutor(max_workers=config["later_round"]["task_parallelism"]) as pool:
        records = list(pool.map(one, task_ids))
    shutil.rmtree(out_dir / "library", ignore_errors=True)
    library.snapshot(out_dir / "library").write_precedents(
        precedents([record["precedent"] for record in records if record["status"] == "precedent"]))
    return records
