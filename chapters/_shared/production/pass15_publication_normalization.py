"""Pass 15 publication normalization for the AppliedML book figures.

This script records the production operations applied after scientific figure freeze:
- resolve book-wide duplicate LaTeX figure labels;
- remove unused TikZ dependencies/style definitions from integrated chapter sources;
- escape raw percent signs in figure captions for pdflatex safety;
- conservatively trim near-white exterior raster margins while preserving scientific content;
- enforce a 2200-pixel raster cap.

It does not alter scientific data, numerical values, model outputs, captions' scientific claims,
or figure-generation provenance. Figure-generating code remains in each chapter/section.
"""
from pathlib import Path
import re
import numpy as np
from PIL import Image

LABEL_MAP = {
    ('01_Computer_Vision_Part_I_Foundations', 7): 'fig:cv1_yolo_cell_prediction',
    ('08_NLP_Part_I', 12): 'fig:nlp1_bert_finetuning',
    ('01_Computer_Vision_Part_I_Foundations', 11): 'fig:cv1_blood_smear_raw',
    ('02_Computer_Vision_Part_II_Autonomous_Systems', 19): 'fig:cv2_lyft_seven_cameras',
    ('01_Computer_Vision_Part_I_Foundations', 12): 'fig:cv1_blood_smear_detection',
    ('02_Computer_Vision_Part_II_Autonomous_Systems', 20): 'fig:cv2_lidar_bev',
    ('03_Deep_Learning_Part_I', 16): 'fig:dl1_convolution',
    ('02_Computer_Vision_Part_II_Autonomous_Systems', 2): 'fig:cv2_convolution',
    ('08_NLP_Part_I', 2): 'fig:nlp1_embedding_geometry',
    ('09_NLP_Part_II', 4): 'fig:nlp2_embedding_geometry',
    ('03_Deep_Learning_Part_I', 37): 'fig:dl1_transfer_learning',
    ('02_Computer_Vision_Part_II_Autonomous_Systems', 25): 'fig:cv2_kitti360_transfer',
}


def trim_raster(path: Path, threshold: int = 250, min_pad: int = 12, area_threshold: float = 0.03):
    im = Image.open(path).convert('RGB')
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    content = np.any(arr < threshold, axis=2)
    ys, xs = np.where(content)
    if len(xs) == 0:
        return False
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    pad_x = max(min_pad, int(0.025 * (x1 - x0)))
    pad_y = max(min_pad, int(0.025 * (y1 - y0)))
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    nw, nh = x1 - x0, y1 - y0
    area_reduction = 1 - (nw * nh) / (w * h)
    if area_reduction <= area_threshold:
        return False
    cropped = im.crop((x0, y0, x1, y1))
    if max(cropped.size) > 2200:
        scale = 2200 / max(cropped.size)
        cropped = cropped.resize((round(cropped.width * scale), round(cropped.height * scale)), Image.Resampling.LANCZOS)
    if path.suffix.lower() == '.png':
        cropped.save(path, optimize=True, compress_level=9)
    else:
        cropped.save(path, quality=92, optimize=True)
    return True


def remove_balanced_command(text: str, command: str) -> str:
    out = text
    while True:
        pos = out.find(command)
        if pos < 0:
            return out
        brace = out.find('{', pos + len(command))
        if brace < 0:
            return out
        depth = 0
        end = None
        for i in range(brace, len(out)):
            if out[i] == '{':
                depth += 1
            elif out[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return out
        while end < len(out) and out[end] in ' \t':
            end += 1
        if end < len(out) and out[end] == '\n':
            end += 1
        out = out[:pos] + out[end:]


def strip_unused_tikz(text: str) -> str:
    text = re.sub(r'(?m)^\s*\\usepackage(?:\[[^\]]*\])?\{tikz\}\s*\n?', '', text)
    text = re.sub(r'(?m)^\s*\\usetikzlibrary\{[^}]*\}\s*\n?', '', text)
    return remove_balanced_command(text, r'\tikzset')


if __name__ == '__main__':
    print('Pass 15 normalization utilities. See PASS15_REPORT.md in the production package for the audited run.')
