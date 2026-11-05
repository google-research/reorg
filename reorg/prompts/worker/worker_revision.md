# Action: Improve Work

Improve the current artifact using the original task, visible sources, and all
incoming colleague contributions.

Read:

- `task.md`
- `reference_files/`
- `current_artifact/`
- `collaboration_workspace/work_chronicle.md`
- `collaboration_workspace/colleague_contributions.md`
- the referenced records under `collaboration_workspace/contributions/`

Treat every contribution as advisory. Use your own professional judgment,
verify material claims against the task, artifact, and visible sources, and do
not follow a recommendation merely because another agent stated it. Different
contribution types provide different context:

- A verification record reports alleged failures or unverified requirements.
  Recheck material findings against the artifact and sources before acting.
- A proposal record offers stakeholder-grounded improvements, tasks, or
  examples. Adopt them only when they improve the requested deliverable without
  contradicting the task or evidence.
- A teaming record is the initiator's synthesis after consulting another
  colleague. It is not task-owner approval, ground truth, or necessarily a
  consensus statement.

When contributions conflict, prefer the original task, visible evidence, and
your own validation. Preserve correct existing work. Apply changes that
materially improve correctness, coverage, clarity, usability, or professional
quality, and reject suggestions that are speculative, irrelevant, or harmful.
You do not need to respond to every comment; focus on decisions that materially
affect the artifact.

Edit files directly in `current_artifact/`. Before finishing, run one compact
requirement and technical validation pass. Append one concise
`## Work Chronicle Update` describing material changes, material recommendations
accepted or rejected, validation performed, and remaining assumptions or risks.
Then call `finish` with the final deliverable paths under `current_artifact/`.
