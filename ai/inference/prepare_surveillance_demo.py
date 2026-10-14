from pathlib import Path
import requests, cv2, json
import numpy as np
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'datasets'/'surveillance_demo'
ROOT.mkdir(parents=True,exist_ok=True)
urls={name:'https://raw.githubusercontent.com/punpayut/Fire-Detection/main/'+name for name in ['fire.mp4','smoke.mp4','small_fire.mp4']}
urls['normal.avi']='https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi'
metadata=[]
canvas=np.full((600,960,3),240,np.uint8)
for i,(name,url) in enumerate(urls.items()):
    dest=ROOT/name
    if not dest.exists():
        r=requests.get(url,timeout=90); r.raise_for_status()
        if r.content.startswith(b'version https://git-lfs.github.com'):
            url=url.replace('raw.githubusercontent.com','media.githubusercontent.com/media')
            r=requests.get(url,timeout=90); r.raise_for_status()
        dest.write_bytes(r.content)
    cap=cv2.VideoCapture(str(dest)); assert cap.isOpened(),name
    fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC,2000); ok,frame=cap.read(); assert ok,name
    tile=cv2.resize(frame,(460,260)); x=(i%2)*480; y=(i//2)*300
    canvas[y:y+260,x:x+460]=tile
    cv2.putText(canvas,name,(x+10,y+285),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
    metadata.append({'file':name,'url':url,'fps':fps,'frames':count,'duration':count/fps})
    cap.release(); print(metadata[-1],flush=True)
cv2.imwrite(str(ROOT/'source_preview.jpg'),canvas)
(ROOT/'sources.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
