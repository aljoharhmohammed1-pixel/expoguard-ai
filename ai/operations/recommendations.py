"""Select and quote approved playbook steps; never invent or execute procedures."""
from datetime import datetime
import hashlib
from typing import Literal
from pydantic import AwareDatetime, Field, StrictBool, model_validator
from ai.core.analysis import Contract, Identifier, Category


class PlaybookStep(Contract):
    stepId: Identifier
    textAr: Identifier
    sourceSection: Identifier


class Playbook(Contract):
    playbookId: Identifier
    version: Identifier
    title: Identifier
    sourceRef: Identifier
    approvalStatus: Literal['DRAFT','APPROVED','REVOKED']
    approvedBy: Identifier | None = None
    approvedAt: AwareDatetime | None = None
    effectiveFrom: AwareDatetime
    expiresAt: AwareDatetime | None = None
    categories: list[Category] = Field(min_length=1)
    zoneIds: list[Identifier] = Field(default_factory=list)
    assetIds: list[Identifier] = Field(default_factory=list)
    requiresKnownAsset: StrictBool = False
    testOnly: StrictBool = False
    steps: list[PlaybookStep] = Field(min_length=1)

    @model_validator(mode='after')
    def valid(self):
        if self.approvalStatus == 'APPROVED' and (self.approvedBy is None or self.approvedAt is None):
            raise ValueError('Approved playbook requires approval attribution')
        if self.expiresAt is not None and self.expiresAt <= self.effectiveFrom:
            raise ValueError('Invalid validity interval')
        if len({s.stepId for s in self.steps}) != len(self.steps):
            raise ValueError('Duplicate step IDs')
        return self


class IncidentFacts(Contract):
    referenceId: Identifier
    category: Category
    evidenceIds: list[Identifier] = Field(min_length=1)
    zoneId: Identifier | None = None
    assetId: Identifier | None = None


class RecommendationRequest(Contract):
    facts: IncidentFacts
    playbooks: list[Playbook] = Field(default_factory=list)
    now: AwareDatetime


class ActionProposal(Contract):
    playbookId: Identifier
    playbookVersion: Identifier
    playbookSha256: Identifier
    sourceRef: Identifier
    sourceSection: Identifier
    stepId: Identifier
    textAr: Identifier
    evidenceIds: list[Identifier] = Field(min_length=1)
    requiresHumanApproval: Literal[True] = True


class RecommendationResult(Contract):
    referenceId: Identifier
    policyVersion: Literal['approved-playbook-v0.1.0'] = 'approved-playbook-v0.1.0'
    status: Literal['PROPOSED','NO_APPROVED_PLAYBOOK','NO_APPLICABLE_PLAYBOOK','AMBIGUOUS_PLAYBOOK']
    proposals: list[ActionProposal] = Field(default_factory=list)
    reasons: list[Identifier]
    requiresHumanReview: Literal[True] = True

    @model_validator(mode='after')
    def consistent(self):
        if (self.status == 'PROPOSED') != bool(self.proposals):
            raise ValueError('Only PROPOSED may contain procedures, and it must contain at least one')
        return self


def recommend(facts, playbooks, *, now: datetime):
    facts = IncidentFacts.model_validate(facts)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must include timezone')
    books = [Playbook.model_validate(book) for book in playbooks]
    if len({(b.playbookId,b.version) for b in books}) != len(books):
        raise ValueError('Duplicate playbook identity/version')
    approved = [b for b in books if b.approvalStatus == 'APPROVED' and not b.testOnly
                and b.approvedAt <= now and b.effectiveFrom <= now and (b.expiresAt is None or now < b.expiresAt)]
    if not approved:
        return RecommendationResult(referenceId=facts.referenceId, status='NO_APPROVED_PLAYBOOK',
            reasons=['No active, approved, non-test playbook; no operational procedure invented.'])
    applicable = [b for b in approved if facts.category in b.categories
                  and (not b.zoneIds or facts.zoneId in b.zoneIds)
                  and (not b.assetIds or facts.assetId in b.assetIds)
                  and (not b.requiresKnownAsset or facts.assetId is not None)]
    if not applicable:
        return RecommendationResult(referenceId=facts.referenceId, status='NO_APPLICABLE_PLAYBOOK',
            reasons=['Approved playbooks do not match the supplied category/zone/asset.'])
    if len(applicable) != 1:
        return RecommendationResult(referenceId=facts.referenceId, status='AMBIGUOUS_PLAYBOOK',
            reasons=['Multiple approved playbooks match; backend/human must select one before procedures are proposed.'])
    book = applicable[0]
    digest = hashlib.sha256(book.model_dump_json().encode('utf-8')).hexdigest()
    return RecommendationResult(referenceId=facts.referenceId, status='PROPOSED',
        reasons=['Exact approved steps retrieved for human approval; no tasks executed.'],
        proposals=[ActionProposal(playbookId=book.playbookId, playbookVersion=book.version, playbookSha256=digest,
            sourceRef=book.sourceRef, sourceSection=step.sourceSection, stepId=step.stepId, textAr=step.textAr,
            evidenceIds=facts.evidenceIds) for step in book.steps])
