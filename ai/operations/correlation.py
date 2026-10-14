"""Explainable local correlation proposals, not incident creation or merging."""
import re
from typing import Literal
from pydantic import AwareDatetime, Field, model_validator
from ai.core.analysis import Contract, Identifier, Category, Score
from ai.text.classifier import normalize

POLICY_VERSION = 'correlation-heuristic-v0.1.0'


class AlertFacts(Contract):
    alertId: Identifier
    category: Category
    observedAt: AwareDatetime
    zoneId: Identifier | None = None
    assetId: Identifier | None = None
    eventId: Identifier | None = None
    summary: str = Field(default='', max_length=20000)
    aiConfidence: Score | None = None


class CorrelationPolicy(Contract):
    windowSeconds: int = Field(default=300, strict=True, gt=0, le=86400)
    reviewThreshold: Score = .7


class CorrelationRequest(Contract):
    left: AlertFacts
    right: AlertFacts
    policy: CorrelationPolicy = Field(default_factory=CorrelationPolicy)


class CorrelationProposal(Contract):
    alertIds: tuple[Identifier, Identifier]
    policyVersion: Identifier = POLICY_VERSION
    windowSeconds: int = Field(gt=0)
    reviewThreshold: Score
    decision: Literal['REVIEW_LINK', 'NO_SUGGESTION', 'SAME_ALERT']
    correlationScore: Score | None
    lexicalSimilarity: Score | None
    timeDeltaSeconds: float = Field(ge=0)
    reasons: list[Identifier]
    requiresHumanReview: Literal[True] = True
    limitations: list[str] = Field(default_factory=lambda: [
        'Heuristic ranking, not calibrated probability or permission to merge.',
        'Lexical overlap is not semantic understanding; backend decides incident membership.'])


def lexical_similarity(left, right):
    """Jaccard overlap after basic Arabic normalization; empty text is unknown."""
    def tokens(text):
        return set(re.findall(r'\w+', normalize(text)[0])) - {'في', 'عند', 'يوجد', 'هناك', 'من', 'على'}
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a and b else None


def correlate(left, right, policy=None):
    left, right = AlertFacts.model_validate(left), AlertFacts.model_validate(right)
    policy = CorrelationPolicy.model_validate(policy or {})
    delta = abs((left.observedAt-right.observedAt).total_seconds())
    base = dict(alertIds=(left.alertId,right.alertId), timeDeltaSeconds=delta,
                windowSeconds=policy.windowSeconds, reviewThreshold=policy.reviewThreshold)
    if left.alertId == right.alertId:
        return CorrelationProposal(**base, decision='SAME_ALERT', correlationScore=None,
                                   lexicalSimilarity=None, reasons=['Same alert identity; not a new link proposal.'])
    conflicts = []
    if left.category != right.category:
        conflicts.append('Different categories.')
    for field in ('eventId','zoneId','assetId'):
        a, b = getattr(left,field), getattr(right,field)
        if a is not None and b is not None and a != b:
            conflicts.append(f'Conflicting trusted {field}.')
    if delta > policy.windowSeconds:
        conflicts.append('Outside configured time window.')
    similarity = lexical_similarity(left.summary, right.summary)
    if conflicts:
        return CorrelationProposal(**base, decision='NO_SUGGESTION', correlationScore=0.0,
                                   lexicalSimilarity=similarity, reasons=conflicts)
    same_zone = left.zoneId is not None and left.zoneId == right.zoneId
    same_asset = left.assetId is not None and left.assetId == right.assetId
    location = same_zone or same_asset
    # Fixed, documented weights; detector confidence is intentionally not consulted.
    score = .2 + (.35 if location else 0) + .25*(1-delta/policy.windowSeconds) + .2*(similarity or 0)
    reasons = ['Same category.', 'Within configured time window.']
    reasons.append('Shared trusted zone or asset.' if location else 'No shared trusted zone or asset; link suggestion withheld.')
    if left.eventId is None or right.eventId is None:
        reasons.append('Event context incomplete.')
    if similarity is None:
        reasons.append('Insufficient text for lexical comparison.')
    return CorrelationProposal(**base, decision='REVIEW_LINK' if location and score >= policy.reviewThreshold else 'NO_SUGGESTION',
                               correlationScore=round(score,6), lexicalSimilarity=similarity, reasons=reasons)


def rank_related(alert, others, policy=None):
    alert = AlertFacts.model_validate(alert)
    others = [AlertFacts.model_validate(value) for value in others]
    if len({value.alertId for value in others}) != len(others):
        raise ValueError('Duplicate alert IDs in candidate list')
    results = [correlate(alert, value, policy) for value in others]
    return sorted(results, key=lambda item: (item.decision != 'REVIEW_LINK', -(item.correlationScore or 0), item.alertIds[1]))
