# Action: Supporting / Proposing

Thoughtfully identify ways to make current artifact more useful from your
social perspective and give the task owner an informative handoff.

Use only these workspace inputs:

- `task.md`
- `reference_files/`
- `current_artifact/`
- `collaboration_workspace/work_chronicle.md`
- `collaboration_workspace/colleague_contributions.md`
- `collaboration_workspace/contributions/`

Inspect the requested deliverables, material task requirements, and intended
recipient closely enough to understand the goals, constraints, decisions, and
experience that matter from your role. Identify a set of unmet needs, missed
opportunities, or weaknesses in usefulness.

Use your social role to prioritize needs and explain why proposed directions
matter. Prior contributions are further context, not artifacts to inspect nor
authority. Use only local inspection tools; do not use web retrieval or edit
the canonical artifact.

Write exactly one publishable file:

```text
collaboration_workspace/improvement_proposal.md
```

Use this structure:

```markdown
# Handoff Summary

# Perspective and Needs

# Identified Gaps

# Prioritized Proposals
```

Keep all four sections nonempty. If there is no meaningful gap or proposal,
explicitly say so in the gap section and propose no unnecessary change.
Distinguish visible evidence from inference. Do not edit the canonical artifact
or any input.

Append one concise `## Work Chronicle Update` to
`collaboration_workspace/work_chronicle.md` summarizing the proposal outcome,
the most important item to carry forward, and important limits.
Do not rewrite prior chronicle content. Then call `finish` with
`collaboration_workspace/improvement_proposal.md`.
