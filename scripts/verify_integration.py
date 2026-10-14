"""Run the actual service locally and capture integration acceptance evidence."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.core.multimodal import UnifiedAnalysis


def main():
    output = ROOT / 'docs/release'
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MODEL_DEVICE='cpu', CAMERA_CONFIG_PATH=str(ROOT/'docs/cameras.example.json'),
               AI_SERVICE_VERSION='0.1.0-integration', PLAYBOOK_DIR=str(ROOT/'playbooks'),
               FIRE_MODEL_PATH=str(ROOT/'ai/models/infernoguard_best.pt'),
               PERSON_MODEL_PATH=str(ROOT/'ai/models/yolo26n.pt'), INCLUDE_PERSON_DETECTIONS='false')
    checks = []
    examples = []
    def record(name, response, request=None):
        payload = response.json()
        checks.append({'name': name, 'httpStatus': response.status_code,
                       'durationMs': response.headers.get('x-process-time-ms'), 'passed': True})
        examples.append({'name': name, 'request': request, 'httpStatus': response.status_code, 'response': payload})
        return payload
    with tempfile.TemporaryFile(mode='w+b') as log:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                                    '--port', str(port), '--no-access-log'], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=120, trust_env=False) as client:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError('Service failed startup; inspect local dependencies/weights')
                    try:
                        response = client.get('/health')
                        if response.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError('Service startup timed out')
                health = record('Health', response)
                assert health['version'] == '0.1.0-integration' and health['modelsReady']
                models = record('Models', client.get('/v1/models'))
                assert all(m['status'] == 'ready' for m in models['models'])
                assert not any('path' in key.lower() for m in models['models'] for key in m)
                swagger = client.get('/docs')
                assert swagger.status_code == 200 and 'swagger-ui' in swagger.text
                checks.append({'name': 'Swagger HTML', 'passed': True, 'httpStatus': 200})
                schema_response = client.get('/openapi.json')
                assert schema_response.status_code == 200
                schema = schema_response.json()
                assert schema['info']['version'] == '0.1.0-integration'
                (ROOT/'docs/openapi.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                checks.append({'name': 'Export OpenAPI from running server', 'passed': True, 'httpStatus': 200})

                def image(name, raw, filename, camera=None, expected='OK'):
                    data = dict(requestId=name, imageId=name, source='CCTV' if camera else 'VISITOR_APP',
                                capturedAt='2026-10-14T09:00:00Z')
                    if camera:
                        data['cameraId'] = camera
                    response = client.post('/v1/analyze/image', data=data, files={'file': (filename, raw, 'image/jpeg')})
                    result = record(name, response, dict(method='POST', path='/v1/analyze/image', form=data, file=filename))
                    assert response.status_code == 200 and result['analysisStatus'] == expected
                    UnifiedAnalysis.model_validate(result)
                    return result
                samples = ROOT/'ai/datasets/fire_smoke_samples'
                fire = image('Fire image', (samples/'fire_1.jpg').read_bytes(), 'fire_1.jpg')
                assert any(d['label'] == 'visible_flame' for d in fire['detections'])
                smoke = image('Smoke image', (samples/'smoke_1.jpg').read_bytes(), 'smoke_1.jpg')
                assert any(d['label'] == 'suspected_smoke' for d in smoke['detections'])
                normal = image('Normal image', (samples/'neutral_1.jpg').read_bytes(), 'neutral_1.jpg')
                assert not normal['detections'] and not normal['candidates']
                image('Corrupt image', b'not an image', 'corrupt.jpg', expected='UNUSABLE')
                import ultralytics
                bus = Path(ultralytics.__file__).parent/'assets/bus.jpg'
                crowd = image('Crowd configured camera', bus.read_bytes(), 'ultralytics-assets-bus.jpg', 'INTEGRATION-DEMO')
                assert crowd['crowdMetrics']['peopleCount'] > 0 and crowd['crowdMetrics']['roiApplied']
                unknown = image('Camera without crowd configuration', bus.read_bytes(), 'ultralytics-assets-bus.jpg', 'UNKNOWN-CAMERA')
                assert 'crowdMetrics' not in unknown and unknown['limitations']
                def post(name, path, data):
                    response = client.post(path, json=data)
                    result = record(name, response, {'method': 'POST', 'path': path, 'json': data})
                    assert response.status_code == 200
                    return result
                text = post('Arabic text', '/v1/analyze/text', {'requestId':'text-demo','textId':'t-1','text':'يوجد دخان عند البوابة'})
                UnifiedAnalysis.model_validate(text)
                assert text['candidates'][0]['category'] == 'FIRE'
                left = dict(alertId='A1', category='FIRE', observedAt='2026-10-14T09:00:00Z', zoneId='DEMO-ZONE')
                right = dict(left, alertId='A2', observedAt='2026-10-14T09:00:08Z')
                correlation = post('Correlation', '/v1/correlation/analyze', {'left':left, 'right':right})
                assert correlation['proposal']['decision'] == 'REVIEW_LINK'
                summary = post('Summary', '/v1/summary', {'alerts':[left, right]})
                assert summary['sourceCount'] == 2
                recommendation = post('Recommendation', '/v1/recommend', {'referenceId':'review-1', 'category':'FIRE', 'evidenceIds':['det-1']})
                assert recommendation['status'] == 'NO_APPROVED_PLAYBOOK' and not recommendation['proposals']
                bad = client.post('/v1/analyze/text', json={})
                assert bad.status_code == 422
                record('Validation error', bad, {'method':'POST', 'path':'/v1/analyze/text','json':{}})
                big = client.post('/v1/analyze/image', content=b'x'*(10*1024*1024+65537), headers={'content-type':'multipart/form-data; boundary=test'})
                assert big.status_code == 413
                record('Oversized body', big)
                wrong = client.post('/v1/analyze/image', data=dict(requestId='mime',imageId='mime',source='VISITOR_APP',capturedAt='2026-10-14T09:00:00Z'), files={'file':('x.txt',b'abc','text/plain')})
                assert wrong.status_code == 415
                record('Unsupported MIME', wrong)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'python':sys.version.split()[0],
              'environment':'clean integration venv', 'device':'cpu', 'passed':True, 'checks':checks}
    (output/'ACCEPTANCE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'API_RESPONSES.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    sections = ['# API examples — v0.1.0-integration\n\nCaptured from the running FastAPI service with real local weights, CPU and the synthetic camera catalog in `docs/cameras.example.json`. No provider responses are simulated. Image filenames identify the uploaded fixtures; request metadata is illustrative. Results are observations, not accuracy measurements.\n\nRun `python scripts/verify_integration.py` to regenerate these examples and OpenAPI. This starts and stops its own temporary local service. Fire/smoke/normal fixtures are in `ai/datasets/fire_smoke_samples`; the people image is bundled with the installed Ultralytics package at `ultralytics/assets/bus.jpg`.\n\nFor a manual upload, set BASE to your service URL and use:\n\n```powershell\n$base = "http://127.0.0.1:8000"\ncurl.exe "$base/v1/analyze/image" -F "file=@ai/datasets/fire_smoke_samples/fire_1.jpg" -F "requestId=demo-1" -F "imageId=img-1" -F "source=VISITOR_APP" -F "capturedAt=2026-10-14T09:00:00Z"\n```\n\nJSON examples below show the exact POST body under `json`; submit it using `Content-Type: application/json` (or paste into Swagger). For example save a body as request.json and run `curl.exe -H "Content-Type: application/json" --data-binary "@request.json" "$base/v1/analyze/text"`. Image `form` fields are multipart fields, not a JSON body.\n']
    for item in examples:
        sections.append('\n## '+item['name']+'\n')
        if item['request'] is not None:
            sections.append('\nRequest:\n```json\n'+json.dumps(item['request'],ensure_ascii=False,indent=2)+'\n```\n')
        sections.append(f'\nResponse: HTTP {item["httpStatus"]}\n```json\n'+json.dumps(item['response'],ensure_ascii=False,indent=2)+'\n```\n')
    (ROOT/'docs/API_EXAMPLES.md').write_text(''.join(sections),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'openapiExported':True}))


if __name__ == '__main__':
    main()
