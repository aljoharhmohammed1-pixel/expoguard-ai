"""HTTP-specific views over the existing pure advisory functions."""
from pydantic import Field
from ai.core.analysis import Contract
from ai.operations.correlation import CorrelationProposal, CorrelationRequest, correlate


class CorrelationResult(Contract):
    proposal: CorrelationProposal
    scoreBreakdown: dict[str, float | bool | None]


def correlation_result(payload: CorrelationRequest):
    proposal = correlate(payload.left, payload.right, payload.policy)
    shared = any(getattr(payload.left,k) is not None and getattr(payload.left,k)==getattr(payload.right,k)
                 for k in ('zoneId','assetId'))
    conflict = payload.left.category != payload.right.category or any(
        getattr(payload.left,k) is not None and getattr(payload.right,k) is not None
        and getattr(payload.left,k)!=getattr(payload.right,k) for k in ('zoneId','assetId','eventId'))
    gated = conflict or proposal.timeDeltaSeconds > payload.policy.windowSeconds
    same = proposal.decision == 'SAME_ALERT'
    return CorrelationResult(proposal=proposal, scoreBreakdown={
        'categoryContribution':None if same else (.2 if payload.left.category==payload.right.category else 0),
        'locationContribution':None if same else (.35 if shared else 0),
        'timeContribution':None if same else .25*max(0,1-proposal.timeDeltaSeconds/payload.policy.windowSeconds),
        'lexicalContribution':None if same else .2*(proposal.lexicalSimilarity or 0),
        'conflictOrTimeGate':gated, 'sameAlert':same, 'finalScore':proposal.correlationScore})
