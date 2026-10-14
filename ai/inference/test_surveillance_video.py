from pathlib import Path
import json, time, subprocess
import argparse
from detection_events import build_detection_events
import cv2
import numpy as np
from ultralytics import YOLO

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'datasets'/'surveillance_demo'
OUT=BASE/'inference'/'runs'/'surveillance_demo'
OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description='CPU video detection with JSON camera events')
parser.add_argument('--zone', default='B1')
parser.add_argument('--camera-id', default='CAM-B1-01')
args=parser.parse_args()
FPS=10
SECONDS=8
scenes=[('normal.avi',0,'Normal A'),('fire.mp4',1,'Fire and smoke A'),('smoke.mp4',1,'Fire and smoke B'),('small_fire.mp4',2,'Fire and smoke C'),('normal.avi',30,'Normal B')]
model=YOLO(str(BASE/'models'/'infernoguard_best.pt'))
assert model.names=={0:'smoke',1:'fire'},model.names
writers=[]
for name in ['input_raw.mp4','detections_raw.mp4']:
    writer=cv2.VideoWriter(str(OUT/name),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(960,600))
    assert writer.isOpened(),name
    writers.append(writer)
records=[]
summaries=[]
previews=[]
events_log=(OUT/'events.jsonl').open('w',encoding='utf-8',buffering=1)
started=time.perf_counter()
try:
    for scene_id,(filename,offset,label) in enumerate(scenes):
        cap=cv2.VideoCapture(str(SRC/filename))
        assert cap.isOpened(),filename
        source_fps=cap.get(cv2.CAP_PROP_FPS)
        summary={'scene':scene_id+1,'label':label,'source':filename,'source_start_seconds':offset,'start_seconds':scene_id*SECONDS,'end_seconds':(scene_id+1)*SECONDS,'frames':0,'fire_frames':0,'smoke_frames':0,'any_detection_frames':0,'max_fire_confidence':0,'max_smoke_confidence':0}
        for j in range(FPS*SECONDS):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round((offset+j/FPS)*source_fps))
            ok,frame=cap.read()
            assert ok,(filename,j)
            h,w=frame.shape[:2]; scale=min(960/w,540/h)
            resized=cv2.resize(frame,(round(w*scale),round(h*scale)))
            fitted=np.zeros((540,960,3),np.uint8)
            top=(540-resized.shape[0])//2; left=(960-resized.shape[1])//2
            fitted[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
            result=model.predict(fitted,device='cpu',conf=.35,imgsz=640,verbose=False)[0]
            detections=[{'class':result.names[int(b.cls.item())],'confidence':round(float(b.conf.item()),4),'xyxy':b.xyxy[0].tolist()} for b in result.boxes]
            for event in build_detection_events(detections, args.zone, args.camera_id):
                line=json.dumps(event, ensure_ascii=False)
                events_log.write(line+'\n')
                print(line, flush=True)
            classes={d['class'] for d in detections}
            summary['frames']+=1
            summary['any_detection_frames']+=bool(classes)
            for cls in ['fire','smoke']:
                summary[cls+'_frames']+=cls in classes
                summary['max_'+cls+'_confidence']=max([summary['max_'+cls+'_confidence']]+[d['confidence'] for d in detections if d['class']==cls])
            t=scene_id*SECONDS+j/FPS
            records.append({'frame':len(records),'time_seconds':round(t,1),'scene':scene_id+1,'detections':detections,'inference_ms':result.speed['inference']})
            status=' + '.join(sorted(classes)).upper() if classes else 'NO DETECTION'
            for k,picture in enumerate([fitted,result.plot(line_width=2,font_size=16)]):
                display=np.zeros((600,960,3),np.uint8); display[:540]=picture
                cv2.putText(display,f'DEMO | CAM {scene_id+1:02d} | {t:04.1f}s | '+('INPUT' if k==0 else status),(16,565),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(display,'CPU | conf 0.35 | 10 fps playback | edited multi-scene footage',(16,590),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
                writers[k].write(display)
                if k==1 and j==FPS*SECONDS//2:
                    previews.append(display.copy())
                    cv2.imwrite(str(OUT/f'scene_{scene_id+1}.jpg'),display)
            if (j+1)%40==0: print(f'PROGRESS scene {scene_id+1}/5 frames {j+1}/80',flush=True)
        cap.release(); summaries.append(summary); print('SCENE',json.dumps(summary),flush=True)
finally:
    events_log.close()
    for writer in writers: writer.release()
elapsed=time.perf_counter()-started
assert len(records)==400,len(records)
import imageio_ffmpeg
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
for source,target in [('input_raw.mp4','input.mp4'),('detections_raw.mp4','detections.mp4')]:
    subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(OUT/source),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/target)],check=True)
    check=cv2.VideoCapture(str(OUT/target)); decoded=0
    while True:
        ok,_=check.read()
        if not ok: break
        decoded+=1
    check.release(); assert decoded==400,(target,decoded)
canvas=np.full((900,960,3),235,np.uint8)
for i,frame in enumerate(previews):
    thumb=cv2.resize(frame,(480,300)); x=(i%2)*480; y=(i//2)*300
    canvas[y:y+300,x:x+480]=thumb
cv2.imwrite(str(OUT/'overview.jpg'),canvas)
report={'model':'infernoguard_best.pt','device':'cpu','confidence':.35,'duration_seconds':40,'playback_fps':10,'frames_processed':len(records),'processing_seconds':round(elapsed,2),'effective_processing_fps':round(len(records)/elapsed,2),'mean_inference_ms':round(sum(r['inference_ms'] for r in records)/len(records),2),'note':'Edited real video excerpts; source labels are descriptive, not frame-level ground truth. Not a formal accuracy benchmark.','sources':json.loads((SRC/'sources.json').read_text()),'scenes':summaries,'frames':records}
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ['frames','sources']}),flush=True)
print('VIDEO:',OUT/'detections.mp4')
print('EVENTS:',OUT/'events.jsonl')

