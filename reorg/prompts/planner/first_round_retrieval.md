Please act as a recommendation system and select communication structures that are likely to help complete the task and produce a high-quality artifact. Each structure defines a message-passing workflow between a central worker and reviewers. Recommend based on the exact handoffs, reviewer perspectives, similarity to summarized supporting tasks, and previous task-specific lessons.

## Current Task for Communication Structure Recommendation

<<TASK>>

## Shared Reviewer Profiles

<<REVIEWER_PROFILES>>

## Available Communication Structures

<<STRUCTURE_CATALOG>>

## Previously Helpful Task Summaries

<<HELPFUL_TASKS>>

## Task-Specific Previous Lessons on Effective Communication Structures and Features

<<LEARNED_INSIGHTS>>

## Instructions

1. Analyze what coordination, review, and revision the current task needs.
2. Compare those needs with the exact stages, direct handoffs, and reviewer profiles in each available graph.
3. Use summarized supporting tasks and previous lessons when relevant. Missing evidence means unknown and does not automatically mean a poor fit.
4. More reviewers or more stages are not automatically better.
5. Recommend exactly <<TOP_K>> available `graph_###` aliases in best-first order.

Return JSON only:

```json
{
  "task_analysis": ["short coordination consideration"],
  "recommendations": [
    {
      "structure_id": "an available graph alias",
      "reason": "brief reason grounded in the current task, exact handoffs, reviewer profile, or learned evidence"
    }
  ]
}
```

`recommendations` must contain exactly <<TOP_K>> unique available graph aliases.
