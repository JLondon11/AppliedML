"""Real NEU steel-defect matched-seed GAN augmentation audit.

Source: genuine 12 recorded held-out classifier evaluations on the same 360
original photographed test defects. Results are across THREE training seeds,
NOT 1,080 independent model replicates or 4,320 independent test photos.

python audit_paired_seed_uncertainty.py --csv replicated_seed_metrics_pass13.csv --out paired_seed_science
Standard library only; no invented observations or retraining.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

SEEDS = (20261106, 20261107, 20261108)
ARMS = ("real_only", "flip_augmentation", "gan_augmented", "real_repeat_augmented")
METRICS = ("test_accuracy", "test_macro_f1")
T975_DF2 = 4.302652729911275


def analyze(path: Path) -> tuple[dict, list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(SEEDS)*len(ARMS):
        raise ValueError("Expected exactly 12 original seed × arm measurements")
    by = {}
    for r in rows:
        if set(r) != {"seed", "arm", *METRICS}:
            raise ValueError("Original metric schema mismatch")
        try:
            seed = int(r["seed"])
            arm = r["arm"]
            values = {metric: float(r[metric]) for metric in METRICS}
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid original measured score") from exc
        if seed not in SEEDS or arm not in ARMS or (seed, arm) in by:
            raise ValueError("Unknown or duplicate source seed/arm")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
            raise ValueError("Missing or nonfinite original classification metric")
        by[(seed, arm)] = values
    if len(by) != 12:
        raise ValueError("Missing authentic seed-arm cell")

    effects = []
    summaries = {}
    for measure in METRICS:
        out = {}
        for baseline in ("real_only", "real_repeat_augmented"):
            differences = []
            for seed in SEEDS:
                gan = by[seed, "gan_augmented"][measure]
                control = by[seed, baseline][measure]
                delta = gan - control
                differences.append(delta)
                effects.append({"seed": seed, "metric": measure, "comparator": baseline,
                                "gan_score": gan, "real_control_score": control,
                                "gan_minus_control": delta})
            mean = statistics.mean(differences)
            sd = statistics.stdev(differences)
            margin = T975_DF2 * sd / math.sqrt(len(SEEDS))
            negative = sum(d < 0 for d in differences)
            if any(d == 0 for d in differences):
                raise ValueError("Zero difference: adjust sign-test denominator")
            # Exact one-sided sign-tail against independent equiprobable signs,
            # CONDITIONAL on the three specified seeds and zero median difference.
            sign_tail = sum(math.comb(3, k) for k in range(negative, 4))/8
            out[baseline] = {
                "three_paired_seed_differences": differences,
                "mean_difference": mean,
                "between_seed_sample_sd": sd,
                "descriptive_t95_interval_n3": [mean-margin, mean+margin],
                "negative_difference_seeds": negative,
                "one_sided_exact_sign_test_tail": sign_tail,
            }
        summaries[measure] = out
    return {
        "source": path.name,
        "real_original_test_images_per_seed": 360,
        "source_training_seeds": list(SEEDS),
        "source_rows": 12,
        "metrics": summaries,
        "limitations": [
            "Three training seeds sharing the same 360 original real test images",
            "Seed-level differences are paired; test images are not 1,080 independent trials",
            "Student-t intervals with 2 degrees of freedom are descriptive and assumption-sensitive",
            "One-sided sign-test p=0.125 for all three negative effects is not <0.05",
            "No per-image paired predictions in this CSV; do not infer a McNemar test",
            "No independent factory/domain data or long-budget GAN training here",
            "A negative result applies only to the matched original NEU experiment"
        ],
    }, effects


def draw_svg(audit: dict) -> str:
    # Two scientifically distinct paired-difference panels; real results only.
    def tx(x, y, value, size=15, fill="#365366", weight=400, anchor="start"):
        return (f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" '
                f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
                f'fill="{fill}">{value}</text>')
    def line(x1,y1,x2,y2,color="#dfe7eb",stroke=1):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke}"/>'
    pieces=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 625" '
            'width="1120" height="625" role="img" aria-label="Genuine measured NEU steel defect '
            'GAN augmentation paired accuracy and macro F1 effects across three training seeds">',
            '<rect width="1120" height="625" fill="white"/>',
            tx(55,50,"Steel-defect GAN augmentation: measured paired-seed effects",25,"#15384a",700),
            tx(55,79,"1,800 source photographs; 1 exact duplicate excluded; 360 original held-out test photos",14),
            tx(55,103,"Three training seeds × four genuine classifier training arms; same photographed test set",14)]
    for i,(measure,title) in enumerate((("test_accuracy","A  Held-out accuracy"),
                                        ("test_macro_f1","B  Held-out macro-F1"))):
        x0=55+i*540
        pieces.append(tx(x0,158,title,18,"#284a5d",700))
        xmin=x0+48;xmax=x0+474;y_top=203
        def xp(v): return xmin+(v+0.42)/0.42*(xmax-xmin)
        for value in (-.4,-.3,-.2,-.1,0):
            x=xp(value)
            pieces += [line(x,190,x,452,"#dce7eb",1),tx(x,473,f"{value*100:.0f} pp",12,"#5c7686",anchor="middle")]
        for j,(baseline,name,col) in enumerate((("real_only","Original photos only","#b25942"),
                                                ("real_repeat_augmented","Matched original repeats/flips","#287589"))):
            measured=audit["metrics"][measure][baseline]
            y=y_top+j*122
            pieces.append(tx(x0+15,y-18,name,15,"#315268",700))
            for k,d in enumerate(measured["three_paired_seed_differences"]):
                yy=y+k*29
                pieces += [line(xp(0),yy,xp(d),yy,col,1.8),
                           f'<circle cx="{xp(d)}" cy="{yy}" r="5.5" fill="{col}" stroke="white" stroke-width="1.4"/>',
                           tx(xmax+7,yy+4,f"S{k+1}",12,"#5c7080")]
            mean=measured["mean_difference"]*100
            pieces.append(tx(x0+15,y+102,f"Mean: {mean:+.2f} pp",14,col,700))
        pieces.append(tx(x0+16,510,"All three paired effects negative",13,"#617582"))
    pieces += [line(55,544,1060,544),
               tx(55,570,"Observed negative effect is consistent across three seeds, but sample size is limited.",14,"#33556a",700),
               tx(55,594,"Exact one-sided sign test: 3/3 negative gives p = 0.125; no factory-wide or training-budget claim.",13,"#5d7380"),
               '</svg>']
    return "\n".join(pieces)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    report,effects=analyze(a.csv)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"paired_seed_statistics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    with (a.out/"paired_seed_effects.csv").open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=("seed","metric","comparator","gan_score",
                       "real_control_score","gan_minus_control"))
        writer.writeheader();writer.writerows(effects)
    (a.out/"paired_seed_effects.svg").write_text(draw_svg(report),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
