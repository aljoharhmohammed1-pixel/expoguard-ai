from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from ai.operations.correlation import AlertFacts, CorrelationRequest, CorrelationProposal, correlate, rank_related
from ai.operations.recommendations import Playbook, IncidentFacts, RecommendationRequest, RecommendationResult, recommend
from ai.handoff import build_readiness, source_digest
from ai.text.classifier import ArabicTextClassifier
from ai.core.multimodal import TextRequest

NOW = datetime(2026,10,13,10,tzinfo=timezone.utc)


def alert(identifier='a', **changes):
    data = dict(alertId=identifier,category='FIRE',observedAt=NOW,zoneId='Z1',eventId='E1',summary='دخان عند البوابة')
    data.update(changes)
    return AlertFacts(**data)


def book(**changes):
    # Synthetic non-emergency procedure; never installed in a production catalogue.
    data = dict(playbookId='TEST-ONLY',version='test-v1',title='Synthetic unit-test document',sourceRef='test-fixture',
        approvalStatus='APPROVED',approvedBy='test-reviewer',approvedAt=NOW-timedelta(days=1),
        effectiveFrom=NOW-timedelta(days=1),categories=['FIRE'],
        steps=[dict(stepId='s1',textAr='مثال اختبار: راجع السجل التجريبي.',sourceSection='test-section-1')])
    data.update(changes)
    return Playbook(**data)


def facts(**changes):
    data = dict(referenceId='backend-reference',category='FIRE',evidenceIds=['request1:det1'],zoneId='Z1')
    data.update(changes)
    return IncidentFacts(**data)


class OperationTests(unittest.TestCase):
    def test_link_proposal_not_merge_and_confidence_independent(self):
        first = correlate(alert(aiConfidence=.1),alert('b',aiConfidence=.99))
        second = correlate(alert(aiConfidence=.99),alert('b',aiConfidence=.1))
        self.assertEqual(first, second)
        self.assertEqual(first.decision,'REVIEW_LINK')
        self.assertNotIn('incidentId',first.model_dump())

    def test_conflicts_and_expired_window(self):
        for changes in [dict(category='CROWD'),dict(zoneId='Z2'),dict(eventId='E2'),dict(observedAt=NOW+timedelta(seconds=301))]:
            self.assertEqual(correlate(alert(),alert('b',**changes)).decision,'NO_SUGGESTION')
        self.assertEqual(correlate(alert(assetId='A1'),alert('b',assetId='A2')).decision,'NO_SUGGESTION')

    def test_missing_location_and_empty_text_not_false_match(self):
        result = correlate(alert(zoneId=None,summary=''),alert('b',zoneId=None,summary=''))
        self.assertEqual(result.decision,'NO_SUGGESTION')
        self.assertIsNone(result.lexicalSimilarity)

    def test_identity_timezone_and_ranking(self):
        self.assertEqual(correlate(alert(),alert()).decision,'SAME_ALERT')
        result = correlate(alert(),alert('b',observedAt=NOW.astimezone(timezone(timedelta(hours=3)))))
        self.assertEqual(result.timeDeltaSeconds,0)
        ranked = rank_related(alert(),[alert('c',zoneId='Z2'),alert('b')])
        self.assertEqual(ranked[0].alertIds[1],'b')
        with self.assertRaises(ValueError):
            rank_related(alert(),[alert('b'),alert('b')])

    def test_exact_approved_text_and_citations(self):
        approved = book()
        result = recommend(facts(),[approved],now=NOW)
        self.assertEqual(result.status,'PROPOSED')
        self.assertEqual(result.proposals[0].textAr,approved.steps[0].textAr)
        self.assertEqual(result.proposals[0].sourceSection,'test-section-1')
        self.assertEqual(result.proposals[0].evidenceIds,['request1:det1'])
        self.assertTrue(result.proposals[0].requiresHumanApproval)

    def test_missing_draft_revoked_test_and_expired_books(self):
        for books in [[],[book(approvalStatus='DRAFT')],[book(approvalStatus='REVOKED')],
                      [book(testOnly=True)],[book(expiresAt=NOW)],
                      [book(approvedAt=NOW+timedelta(days=1))],[book(effectiveFrom=NOW+timedelta(days=1))]]:
            result = recommend(facts(),books,now=NOW)
            self.assertEqual(result.status,'NO_APPROVED_PLAYBOOK')
            self.assertEqual(result.proposals,[])

    def test_playbook_scope_and_ambiguity(self):
        for b in [book(categories=['CROWD']),book(zoneIds=['Z2']),book(assetIds=['A1']),book(requiresKnownAsset=True)]:
            self.assertEqual(recommend(facts(),[b],now=NOW).status,'NO_APPLICABLE_PLAYBOOK')
        self.assertEqual(recommend(facts(),[book(),book(version='test-v2')],now=NOW).status,'AMBIGUOUS_PLAYBOOK')

    def test_invalid_approval_and_time(self):
        with self.assertRaises(ValidationError):
            book(approvedBy=None)
        with self.assertRaises(ValueError):
            recommend(facts(),[book()],now=NOW.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            recommend(facts(),[book(),book()],now=NOW)

    def test_pipeline_text_to_backend_facts_to_advisory_only(self):
        analysis = ArabicTextClassifier().analyze(TextRequest(requestId='r1',textId='t1',text='يوجد دخان',zoneId='Z1'))
        candidate = analysis.candidates[0]
        supplied_by_backend = AlertFacts(alertId='backend-a',category=candidate.category,observedAt=NOW,
            zoneId=candidate.zoneId,summary=analysis.detections[0].evidence,aiConfidence=candidate.confidence)
        self.assertEqual(correlate(supplied_by_backend,alert('b')).decision,'REVIEW_LINK')
        result = recommend(facts(evidenceIds=[f'{analysis.requestId}:{d.detectionId}' for d in analysis.detections]),[],now=NOW)
        self.assertEqual(result.status,'NO_APPROVED_PLAYBOOK')

    def test_operational_schemas_current(self):
        root = Path(__file__).resolve().parents[1]/'contracts'
        for name, model in [('correlation-request',CorrelationRequest),('correlation-proposal',CorrelationProposal),
                            ('playbook',Playbook),('recommendation-request',RecommendationRequest),('recommendation-result',RecommendationResult)]:
            self.assertEqual(json.loads((root/f'{name}.schema.json').read_text(encoding='utf-8')),model.model_json_schema())

    def test_stale_or_fast_test_report_cannot_claim_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            for values in [dict(sourceDigest='old',integration=True),dict(sourceDigest=source_digest(),integration=False)]:
                path.write_text(json.dumps(dict(passed=True,skipped=0,testsRun=1,errors=[],failures=[],**values)),encoding='utf-8')
                result = build_readiness(path)
                self.assertFalse(result['localIntegrationTestsPassed'])
                self.assertFalse(result['productionReady'])
                self.assertEqual(result['status'],'LOCAL_CHECKS_INCOMPLETE')
