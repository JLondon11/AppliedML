"""Shared utilities for the Pass 14 source/provenance closure.

These helpers do not invent empirical values.  They either download a declared
public source, extract the artwork associated with a caption, or hash an output
produced by a chapter-specific executable experiment.
"""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin
import hashlib, io, json, re
import requests
from bs4 import BeautifulSoup
from PIL import Image

UA = {"User-Agent": "AppliedML-book-production/1.0 (+https://github.com/JLondon11/AppliedML)"}


def fetch(url: str, timeout: int = 90) -> bytes:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_provenance(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def crop_white(path: Path, pad: int = 8) -> None:
    """Crop only near-white exterior whitespace; retain all scientific content."""
    im = Image.open(path).convert("RGB")
    import numpy as np
    a = np.asarray(im)
    mask = np.any(a < 248, axis=2)
    if not mask.any():
        return
    ys, xs = np.where(mask)
    x0, x1 = max(0, xs.min()-pad), min(im.width, xs.max()+pad+1)
    y0, y1 = max(0, ys.min()-pad), min(im.height, ys.max()+pad+1)
    im.crop((x0,y0,x1,y1)).save(path, optimize=True)


def extract_html_figure(page_url: str, caption_fragment: str, out_path: Path) -> dict:
    """Extract an image from a public HTML article by matching caption text."""
    html = fetch(page_url)
    soup = BeautifulSoup(html, "html.parser")
    target = None
    needle = re.sub(r"\s+", " ", caption_fragment.lower()).strip()
    for fig in soup.find_all(["figure", "div"]):
        txt = re.sub(r"\s+", " ", fig.get_text(" ", strip=True).lower())
        if needle in txt and fig.find("img"):
            target = fig
            break
    if target is None:
        # PMC often uses fig-group wrappers and links to full-size images.
        for cap in soup.find_all(["figcaption", "p", "div"]):
            txt = re.sub(r"\s+", " ", cap.get_text(" ", strip=True).lower())
            if needle in txt:
                parent = cap
                for _ in range(5):
                    if parent is None: break
                    img = parent.find("img")
                    if img:
                        target = parent; break
                    parent = parent.parent
                if target is not None: break
    if target is None:
        raise RuntimeError(f"caption fragment not found: {caption_fragment}")
    img = target.find("img")
    src = img.get("data-src") or img.get("data-original") or img.get("src")
    if not src:
        raise RuntimeError("matched figure has no image source")
    img_url = urljoin(page_url, src)
    data = fetch(img_url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve source image exactly when already PNG/JPEG; Pillow normalizes otherwise.
    try:
        Image.open(io.BytesIO(data)).convert("RGB").save(out_path, dpi=(300,300), optimize=True)
    except Exception:
        write_bytes(out_path, data)
    crop_white(out_path)
    return {"page_url": page_url, "image_url": img_url, "source_sha256": sha256_bytes(data), "caption_fragment": caption_fragment}


def extract_pdf_figure_near_caption(pdf_url: str, caption_fragment: str, out_path: Path, band_points: float = 380.0) -> dict:
    """Rasterize the visual region immediately above a verified caption in a PDF.

    The crop is chosen from the caption y-coordinate upward, stopping at either the
    preceding caption or `band_points`.  This avoids including the printed caption
    itself in the production artwork.
    """
    import fitz
    data = fetch(pdf_url)
    doc = fitz.open(stream=data, filetype="pdf")
    needle = re.sub(r"\s+", " ", caption_fragment).strip().lower()
    for pno, page in enumerate(doc):
        text = re.sub(r"\s+", " ", page.get_text("text")).lower()
        if needle not in text:
            continue
        words = page.get_text("words")
        # Find first distinctive 2-4 words of caption.
        toks = [t for t in re.findall(r"[A-Za-z0-9-]+", caption_fragment) if len(t)>2][:4]
        rects=[]
        for tok in toks:
            rects += page.search_for(tok)
        if not rects:
            continue
        cy = max(r.y0 for r in rects)
        y0 = max(0.0, cy - band_points)
        # Exclude caption line itself.
        clip = fitz.Rect(0, y0, page.rect.width, max(y0+20, cy-5))
        pix = page.get_pixmap(matrix=fitz.Matrix(2.4,2.4), clip=clip, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        crop_white(out_path)
        return {"pdf_url": pdf_url, "pdf_sha256": sha256_bytes(data), "page": pno+1, "caption_fragment": caption_fragment, "clip": [clip.x0,clip.y0,clip.x1,clip.y1]}
    raise RuntimeError(f"caption not located in PDF: {caption_fragment}")


def copy_repo_asset(src: Path, dst: Path) -> dict:
    data = src.read_bytes()
    write_bytes(dst, data)
    crop_white(dst)
    return {"repo_source": str(src), "source_sha256": sha256_bytes(data)}
