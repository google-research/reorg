# Recommend useful communication for the next round

Please recommend communication flows that are likely to help improve the
current work. Also rank the option to leave the work unchanged.

Your job is to rank the options. A separate rule will decide whether another
round of work actually happens.

## Current work and discussion so far

<<CURRENT_WORK_SUMMARY>>

## People who may take part in a review

<<REVIEWER_PROFILES>>

## Available communication flows

Flows that were already used, along with flows containing no communication,
have already been removed.

<<AVAILABLE_COMMUNICATION_FLOWS>>

## Examples from similar work states

Each example describes one earlier work state and what happened when different
communication flows were tried. Treat it as a limited observation, not a rule
that always applies.

<<PAST_STATE_EXAMPLES>>

## Options

- `leave_unchanged`: keep the current work exactly as it is, with no additional
  reviewer or owner call.
- `flow_###`: run that communication flow, then let the owner revise the work
  once using the resulting advice.

## Instructions

1. Start from the summary. Consider what the chronicle says was done, what
   remains unresolved, what earlier advice was already used, and the risk of
   disturbing good work.
2. Compare the remaining needs with the exact reviewers, actions, order, and
   handoffs in each available communication flow.
3. Use earlier examples only when their work state is meaningfully similar.
   Missing evidence does not mean that a flow is poor.
4. Rank `leave_unchanged` highly when the work appears complete, remaining
   concerns are weak or unverified, or another revision presents substantial
   risk.
5. More reviewers, stages, or messages are not automatically better. Prefer a
   focused flow when the remaining needs are narrow.
6. Recommend exactly nine unique available `flow_###` options and include
   `leave_unchanged` exactly once, for ten options total. Do not invent IDs.
7. Briefly explain your overall reasoning in `analysis`, then return the
   options in best-first order. The list order is the ranking.
8. Give a short reason for every option based on the current work, the people
   and handoffs in the flow, or a relevant earlier example.
9. Each list item should contain only `option_id` and `reason`.
10. Do not output a separate yes/no decision, a threshold, a calculated score,
    or an expected absolute quality score.

Return JSON only, matching the supplied schema.
