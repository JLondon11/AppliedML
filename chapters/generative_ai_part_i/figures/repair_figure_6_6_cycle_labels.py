"""Precise Figure 6.6 diagram annotation repair (no experimental data changed).

Input: source figure path, output: updated figure with mathematically correct
CycleGAN image-domain reconstruction labels and complete bottom panel borders.
The example photographs are schematics, not outputs of an executed benchmark.
"""
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import argparse
p=argparse.ArgumentParser()
p.add_argument("--input",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
im=Image.open(a.input).convert("RGB")
if im.size!=(2020,505):
    raise ValueError(f"Expected original schematic geometry 2020x505, not {im.size}")
new=Image.new("RGB",(2020,540),"white");new.paste(im,(0,0))
draw=ImageDraw.Draw(new)
draw.rectangle((1100,464,1430,505),fill="white")
draw.rectangle((1680,464,1984,505),fill="white")
font=ImageFont.truetype("/usr/share/fonts/opentype/freefont/FreeSerif.otf",31)
for expr,xcenter in [("F(G(x)) ≈ x",1227),("G(F(y)) ≈ y",1825)]:
    bounds=draw.textbbox((0,0),expr,font=font)
    draw.text((xcenter-(bounds[2]-bounds[0])/2,468),expr,font=font,fill=(18,18,18))
ink=(20,23,26)
for left,right in [(14,1018),(1036,2005)]:
    draw.line((left,9,left,528),fill=ink,width=2)
    draw.line((right,9,right,528),fill=ink,width=2)
    draw.line((left,528,right,528),fill=ink,width=2)
a.output.parent.mkdir(parents=True,exist_ok=True)
new.save(a.output,optimize=True)
print("Corrected domain labels: horse F(G(x))~x; zebra G(F(y))~y")
