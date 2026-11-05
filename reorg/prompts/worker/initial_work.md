# Action: Initial Work

Create the first complete, usable version of every deliverable requested by the
task.

Use `task.md` and `reference_files/` as the source of truth. Treat
`current_artifact/` as the artifact root. Create final deliverable files there;
place temporary scripts or intermediate files under `scratch/` only.

Inspect all task requirements and reference files needed to complete the work.
Make reasonable, explicit assumptions when an exact relationship cannot be
resolved from the visible sources. Do not wait for colleague input: this action
establishes the artifact that later actions will inspect.

Before finishing:

1. Confirm every requested deliverable and filename exists.
2. Open or programmatically validate each file when practical.
3. Check that the deliverables are populated, usable, and in the requested
   formats.

Append one concise `## Work Chronicle Update` to
`collaboration_workspace/work_chronicle.md` describing what you created,
important assumptions, and validation performed. Then call `finish` with the
relative deliverable paths under `current_artifact/`.
