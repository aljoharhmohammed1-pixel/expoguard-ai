from pathlib import Path
import json
import time
import requests
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'datasets' / 'fire_smoke_samples'
INPUT.mkdir(parents=True, exist_ok=True)
SOURCE = 'https://raw.githubusercontent.com/DeepQuestAI/Fire-Smoke-Dataset/master/Assets/'
names = ['fire_1.jpg', 'fire_2.jpg', 'smoke_1.jpg', 'smoke_2.jpg', 'neutral_1.jpg', 'neutral_2.jpg']
manifest = []
for name in names:
    path = INPUT / name
    if not path.exists():
        response = requests.get(SOURCE + name, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    with Image.open(path) as im:
        im.verify()
    manifest.append({'file': name, 'url': SOURCE + name, 'source_category': name.split('_')[0]})
(INPUT / 'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
model = YOLO(str(BASE / 'models' / 'infernoguard_best.pt'))
records = []
output = None
for name in names:
    result = model.predict(source=str(INPUT / name), device='cpu', imgsz=640, conf=0.35,
                           save=True, project=str(BASE / 'inference' / 'runs'),
                           name='fire_smoke_samples', exist_ok=True)[0]
    output = Path(result.save_dir)
    detections = [{'class': result.names[int(box.cls.item())], 'confidence': round(float(box.conf.item()), 4),
                   'xyxy': box.xyxy[0].tolist()} for box in result.boxes]
    records.append({'file': name, 'source_category': name.split('_')[0], 'detections': detections,
                    'speed_ms': result.speed})
    print('RESULT', json.dumps(records[-1]), flush=True)
report = {'model': 'infernoguard_best.pt', 'device': 'cpu', 'confidence_threshold': 0.35,
          'image_size': 640, 'source': 'https://github.com/DeepQuestAI/Fire-Smoke-Dataset',
          'note': 'Small demonstration sample, not an independent accuracy benchmark; training overlap unknown.',
          'results': records}
(output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
canvas = Image.new('RGB', (1200, 1140), '#f4f6f8')
draw = ImageDraw.Draw(canvas)
for i, row in enumerate(records):
    x, y = (i % 2) * 600, (i // 2) * 380
    with Image.open(output / row['file']) as im:
        tile = ImageOps.contain(im.convert('RGB'), (580, 315))
        canvas.paste(tile, (x + (600 - tile.width)//2, y + 10 + (315 - tile.height)//2))
    label = row['file'] + ' | ' + (', '.join(f"{d['class']} {d['confidence']:.0%}" for d in row['detections']) or 'No detections')
    draw.text((x + 15, y + 337), label, fill='#111827', font_size=18)
canvas.save(output / 'overview.jpg', quality=94)
print('OVERVIEW:', output / 'overview.jpg')
print('REPORT:', output / 'report.json')
