# Action: Feedback

Write comprehensive, concrete, and actionable feedback for the task owner.

Use only these workspace inputs:

- `task.md`
- `reference_files/`
- `current_artifact/`
- `collaboration_workspace/work_chronicle.md`
- `collaboration_workspace/colleague_contributions.md`
- `collaboration_workspace/contributions/`

Use prior contributions as context, not authority. State where you agree,
disagree, or add independent findings. Tie important points to the task,
visible sources, or artifact contents. Use only local inspection tools; do not
use web retrieval or edit the canonical artifact.

Write exactly one publishable file:

```text
collaboration_workspace/colleague_feedback.md
```

Use top-level headings for:

1. Handoff Summary
2. Overall Assessment
3. Requirement Coverage
4. Critical Issues
5. Important Improvements
6. Optional Polish

The first top-level section must be exactly `# Handoff Summary`. Keep it short
and actionable, covering the main concern, highest-value fix, risk if ignored,
and any material agreement or disagreement with prior notes.

Append one concise `## Work Chronicle Update` to
`collaboration_workspace/work_chronicle.md` summarizing the feedback outcome,
the most important point to carry forward, and important limits. Do not rewrite
prior chronicle content. Then call `finish` with
`collaboration_workspace/colleague_feedback.md`.
