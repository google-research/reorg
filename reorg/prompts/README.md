# Prompt templates

The prompt templates and stakeholder profiles. `manifest.json` stores each file's sha256, and `tests/test_core.py` checks the files against it. The paper's appendix shows excerpts (boxes P1–P11) and the complete stakeholder profiles (R1–R10).

In the model-facing wording, "task owner" is the worker $W$, "colleague" is a stakeholder, "communication structure" (or "flow") is a reflection skill, and "steering" is reflection.

| File | Paper | Used by |
|---|---|---|
| `worker/worker_role.md` | P1 | worker role instructions, every worker call |
| `worker/initial_work.md` | P1 | initial artifact $A_0$ |
| `worker/worker_revision.md` | P1 | revision $A_r$ after contributions $F_r$ |
| `actions/feedback.md` | P2 | Feedback action |
| `actions/propose.md` | P3 | Propose action |
| `loop_in/loop_in.md` | P4 | Loop-In action |
| `loop_in/select.md` | P4 | Loop-In: partner selection and agenda |
| `loop_in/discussion.md` | P4 | Loop-In: one discussion turn |
| `loop_in/summary.md` | P4 | Loop-In: initiator synthesis |
| `roles/*.md` | R1–R10 | stakeholder profiles, combined with an action prompt |
| `planner/first_round_retrieval.md` | P5 | first-round skill retrieval |
| `learning/artifact_ranking.md` (+ `.schema.json`) | P6 | label-free blinded artifact ranking |
| `learning/lesson_compaction.md` | P7 | Summarize: retrieval lessons for $M$ |
| `planner/state_projection.md` (+ `.schema.json`) | P8 | work-state projection |
| `planner/completion_assessment.md` (+ `.schema.json`) | P9 | completion gate |
| `planner/later_round_retrieval.md` (+ `.schema.json`) | P10 | later-round skill retrieval |
| `learning/later_round_lesson.md` (+ `.schema.json`) | P11 | state-conditioned precedents for $M$ |

Role files and paper profiles:

| Role ID | Paper |
|---|---|
| `same_job_teammate` | R1 |
| `direct_manager` | R2 |
| `upstream_requester` | R3 |
| `cross_functional_partner` | R4 |
| `technical_customer` | R5 |
| `implementation_partner` | R6 |
| `procurement_evaluator` | R7 |
| `legal_risk_reviewer` | R8 |
| `internal_auditor` | R9 |
| `experienced_mentor` | R10 |
