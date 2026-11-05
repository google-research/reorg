# Judge current task completion

Estimate how complete the current work is relative to the original task. Use
only the task and work summary below. Judge completion of the requested work,
not whether an optional reviewer could always suggest something else.

Return one integer score from 1 to 10. Use the full scale when appropriate:

- 1: the task is incomplete or there is no usable deliverable; steering is
  very likely to help.
- 2-3: major required work is missing or unresolved.
- 4-5: the work is partial; important requirements or decisions remain.
- 6-7: a core deliverable exists, but substantive gaps or uncertainty remain.
- 8: most requirements appear covered, but at least one meaningful check or
  improvement remains.
- 9: the work is nearly ready; only small gaps or final validation remain.
- 10: the task is mostly done and should stop here; another steering round is
  unlikely to add enough value.

Distinguish among intermediate cases instead of defaulting to an endpoint. Base
the score on the evidence in this state, and briefly explain which completed or
missing requirements determine it.

## Current task and work state

<<CURRENT_WORK_SUMMARY>>

Return JSON with exactly these fields:

```json
{
  "analysis": "brief evidence-based explanation",
  "task_completion_score": 1
}
```
