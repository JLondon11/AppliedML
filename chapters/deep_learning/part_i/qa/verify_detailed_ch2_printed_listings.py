"""Verify the printed Python in the detailed Chapter 2 LaTeX source.

Usage: python verify_detailed_ch2_printed_listings.py Deep_Learning_Part_I_InProgress.tex
This validates structural integrity and syntax, not model execution.
"""
import argparse
import ast
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r'\\begin\{lstlisting\}(?:\[(.*?)\])?\s*\n(.*?)\\end\{lstlisting\}',
    re.S,
)

def verify(filename):
    source = Path(filename).read_text(encoding="utf-8")
    blocks = list(PATTERN.finditer(source))
    assert len(blocks) == 111, f"Expected 111 listings, got {len(blocks)}"
    labels = []
    lengths = []
    legacy = []
    for n, match in enumerate(blocks, 1):
        options = match.group(1) or ""
        label_match = re.search(r"label=\{([^}]*)\}", options)
        label = label_match.group(1) if label_match else f"UNLABELED_{n}"
        code = match.group(2)
        ast.parse(code, filename=label)
        labels.append(label)
        lengths.append(len(code.splitlines()))
        if any(token in code for token in
               ("K.placeholder", "K.gradients", "backend.variable", "fmin_l_bfgs_b")):
            legacy.append(label)
    assert len(set(labels)) == len(labels), "Duplicate code-listing label"
    assert max(lengths) <= 25, "Oversized printed code fragment"
    assert len(re.findall(r"\\begin\{figure\}", source)) == 73
    assert len(re.findall(r"\\begin\{table\}", source)) == 8
    temporal = next(m.group(2) for m in blocks
                    if "lst:dl1_har_lstm_model" in (m.group(1) or ""))
    training = next(m.group(2) for m in blocks
                    if "lst:dl1_har_compile_fit" in (m.group(1) or ""))
    assert "shape=(128, 9)" in temporal and "Inertial Signals" in temporal
    assert "GroupShuffleSplit" in training
    assert "sparse_categorical_crossentropy" in training
    return {
        "syntax_pass": len(blocks),
        "maximum_listing_lines": max(lengths),
        "figure_environments": 73,
        "substantive_tables": 8,
        "legacy_runtime_holds": legacy,
        "scientific_model_execution_verified": False,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("latex_source")
    print(json.dumps(verify(parser.parse_args().latex_source), indent=2))
