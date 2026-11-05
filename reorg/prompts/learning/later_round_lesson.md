# Summarize what communication was useful for this work

Act as a careful note taker. Several communication approaches were applied to
the same current work, and each was followed by one normal Task Owner revision.
The resulting versions were compared with leaving the current work unchanged.
Write a useful precedent for deciding how people should communicate on future
work in a genuinely similar state.

## Task

<<TASK>>

## Current work before the tested communication

<<CURRENT_WORK_SUMMARY>>

## Communication approaches that were tried

The descriptions below state who participated, what each person did, which
activities happened in parallel or in sequence, and how their work reached the
Task Owner. Leaving the work unchanged means no additional reviewer or owner
call.

<<COMMUNICATION_FLOWS>>

## Comparison results and judge observations

<<COMPARISON_RESULTS>>

## Calculated summary

<<CALCULATED_SUMMARY>>

## Limits of the evidence

<<EVIDENCE_LIMITS>>

## Instructions

1. Compare results only within this task and the versions shown here. A
   communication approach that ranked first is not necessarily good in
   absolute terms, and this evidence does not establish causality.
2. In `helpful_communication_flows_if_steering`, describe the complete
   communication approaches that ranked above or below leaving the work
   unchanged. Identify them in ordinary language through their social roles,
   actions, ordering, and handoffs—not through temporary names or the number of
   people involved.
3. In `helpful_communication_flow_features_if_steering`, identify the specific
   roles, review actions, sequences, parallel branches, or handoffs that
   coincided with better or worse results. Compare shared and distinguishing
   features across approaches before making a feature claim.
4. Use the judge observations to connect communication to what changed in the
   resulting versions. Do not describe an issue already completed in the
   current work as a remaining need.
5. Do not infer that more reviewers, more discussion, or a larger flow is
   better merely because a larger approach ranked highly. Prefer the narrowest
   claim supported by contrasts among the tested approaches.
6. Preserve ties, conflicting judge interpretations, missing versions, failed
   runs, and uncertainty from having only one resulting version per approach.
   When judges disagree about what the task required, state that disagreement
   rather than choosing a side.
7. Use cautious language such as “in this run,” “ranked above,” “coincided
   with,” or “may have helped.” Do not claim that an approach or feature is
   always best, necessary, or guaranteed to help.
8. Do not expose temporary labels, internal IDs, fingerprints, file paths, or
   ranking-system mechanics in the output. Describe the communication as a
   human collaborator would.
9. Do not discuss other training tasks. This note will become one independent
   example indexed by the current work state.

Return JSON only, matching the supplied schema.
