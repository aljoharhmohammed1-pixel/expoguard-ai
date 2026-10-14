"""Run from repository root: python -m ai.contracts.export_schema."""
import json
from pathlib import Path
from ai.core.analysis import ImageRequest, VisionAnalysis
from ai.core.multimodal import UnifiedAnalysis, TextRequest, AudioRequest, Transcript
from ai.vision.general import VisionResponse
from ai.text.classifier import TextResponse
from ai.operations.correlation import CorrelationRequest, CorrelationProposal
from ai.operations.recommendations import RecommendationRequest, RecommendationResult, Playbook
from ai.integration.service import IntegrationRequest, IntegrationResponse
from ai.crowd.config import CameraCatalog
from ai.crowd.schemas import CrowdMetrics


def main():
    destination = Path(__file__).resolve().parent
    for name, model in [('image-request', ImageRequest), ('vision-analysis', VisionAnalysis),
                        ('unified-analysis-v2', UnifiedAnalysis), ('text-request', TextRequest),
                        ('audio-request', AudioRequest), ('transcript', Transcript), ('vision-provider-response', VisionResponse),
                        ('text-provider-response', TextResponse), ('correlation-request', CorrelationRequest),
                        ('correlation-proposal', CorrelationProposal), ('playbook', Playbook),
                        ('recommendation-request', RecommendationRequest), ('recommendation-result', RecommendationResult),
                        ('integration-request', IntegrationRequest), ('integration-response', IntegrationResponse),
                        ('camera-catalog', CameraCatalog), ('crowd-metrics', CrowdMetrics)]:
        (destination / f'{name}.schema.json').write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
