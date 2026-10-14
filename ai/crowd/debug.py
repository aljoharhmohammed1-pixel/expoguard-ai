"""Offline developer visualization; never part of the production response."""
import argparse
from pathlib import Path
from PIL import ImageDraw
from ai.crowd.geometry import bottom_center, point_in_polygon
from ai.crowd.config import load_cameras
from ai.crowd.person import PersonDetector
from ai.crowd.analysis import measure, skip_reason
from ai.quality.image_quality import quality_check


def render_debug(image, people, camera, metrics):
    output = image.copy().convert('RGB')
    draw = ImageDraw.Draw(output)
    width, height = output.size
    if camera.crowd.roi:
        polygon = [(x*(width-1), y*(height-1)) for x,y in camera.crowd.roi]
        draw.line(polygon+[polygon[0]], fill='yellow', width=3)
    for person in people:
        x,y = bottom_center(person.bbox)
        inside = camera.crowd.useFullFrame or point_in_polygon((x,y),camera.crowd.roi)
        color = 'lime' if inside else 'gray'
        draw.rectangle(tuple(value*scale for value,scale in zip(person.bbox,(width,height,width,height))), outline=color,width=2)
        px,py = x*(width-1),y*(height-1)
        draw.ellipse((px-4,py-4,px+4,py+4),fill=color)
    text = f'count={metrics.peopleCount} capacity={metrics.capacity} ratio={metrics.occupancyRatio} level={metrics.crowdLevel}'
    draw.rectangle((0,0,width,24),fill='black')
    draw.text((5,5),text,fill='white')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--camera',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('ai/models/yolo26n.pt'))
    args = parser.parse_args()
    camera = load_cameras(args.config).resolve(args.camera)
    if skip_reason(camera):
        parser.error(skip_reason(camera))
    image,quality = quality_check(args.image.read_bytes())
    if not quality.usable:
        parser.error('Image failed quality validation')
    detector = PersonDetector(args.model)
    detector.start()
    people = detector.detect(image,image_size=camera.crowd.inferenceImageSize)
    metrics,_,_ = measure(people,camera)
    render_debug(image,people,camera,metrics).save(args.output)
    print(metrics.model_dump_json())


if __name__ == '__main__':
    main()
