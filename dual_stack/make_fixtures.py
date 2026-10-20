#!/usr/bin/env python3
"""Deterministic synthetic image fixtures, not a robot-task accuracy dataset."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw


def main():
    root = Path(__file__).with_name('fixtures')
    root.mkdir(exist_ok=True)
    records = []
    for index in range(2):
        image = Image.new('RGB', (512,384), '#eee9df')
        d = ImageDraw.Draw(image)
        d.polygon([(0,180),(512,180),(512,384),(0,384)], fill='#c3a57e')
        # Cup, cube and ball; no text labels that would make this an OCR test.
        colors = ['#c73232','#2e71c5','#37a55b'] if index == 0 else ['#2e71c5','#c73232','#e2b72e']
        d.ellipse((55,275,155,298), fill='#a98c65')
        d.rectangle((64,199,142,273), fill=colors[0])
        d.ellipse((64,258,142,286), fill=colors[0])
        d.ellipse((64,187,142,213), fill='#edc6b9')
        d.ellipse((72,191,134,207), fill='#5e3632')
        d.arc((125,214,169,256),270,90,fill=colors[0],width=9)
        d.polygon([(220,228),(267,206),(311,230),(263,254)],fill='#80a9d8')
        d.polygon([(220,228),(263,254),(263,303),(220,276)],fill=colors[1])
        d.polygon([(263,254),(311,230),(311,279),(263,303)],fill='#24518a' if index==0 else '#8c2222')
        d.ellipse((365,246,447,328),fill=colors[2])
        d.ellipse((378,255,398,272),fill='#b1dba7')
        path=root/f'scene-{index}.png'
        image.save(path)
        records.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'size':[512,384],'purpose':'synthetic table objects; performance and visual smoke test only'})
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
