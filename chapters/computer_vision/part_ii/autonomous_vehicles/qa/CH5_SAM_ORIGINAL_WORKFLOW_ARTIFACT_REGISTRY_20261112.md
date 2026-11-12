# Chapter 5 — verified original SAM and Mask R-CNN workflow artifact registry

Checked against the connected GitHub Actions artifact listing on 2026-11-12. These are **original recorded model-inference artifacts**, not replacement numerical fixtures. The listings establish existence, byte size and current expiry status; their ZIP contents remain to be independently inspected before complete R9 certification.

| Experiment | Workflow run | Artifact ID | Original artifact name | Size (bytes) | Current status | Expires at (UTC) |
|---|---:|---:|---|---:|---|---|
| SAM ViT-B on 18 original COCO scenes / 59 objects | [41980537230](https://github.com/JLondon11/AppliedML/actions/runs/41980537230) | 13394195846 | `chapter5-expanded-SAM-ViTB-COCO-real-inference` | 36,239,351 | unexpired | 2027-02-03 06:27:29 |
| SAM ViT-B on 24 original COCO scenes / 88 objects | [41981250063](https://github.com/JLondon11/AppliedML/actions/runs/41981250063) | 13393449670 | `chapter5-original-SAM-ViTB-real-COCO-inference` | 28,593,183 | unexpired | 2027-02-03 06:33:52 |
| Matched torchvision Mask R-CNN on 18 original COCO scenes / 59 annotation IDs | [41981859101](https://github.com/JLondon11/AppliedML/actions/runs/41981859101) | 13394521613 | `chapter5-maskrcnn-COCO-matched-18-images` | 40,049,948 | unexpired | 2027-02-03 06:39:20 |

**Required independent inspection before R9 PASS:** download all three original ZIP artifacts, retain file/ZIP SHA256 in durable repository-approved storage, list source images/GT masks/per-instance prediction/metric files and their SHA256 values, verify 18+24 disjoint source image IDs, match 147 recorded original masks and 99 valid paired GrabCut objects to aggregate data, independently regenerate measured scientific figures from exact original per-item data, and compare with the latest Chapter 5 manuscript's exact figures/captions and printed PDF. Confirm source GitHub code and model checkpoint hashes against original run manifests.

**Important:** these original workflow artifacts have expiration dates. Their presence in the GitHub API now does **not** establish permanent source publication; ensure durable legal storage or verified retrieval before expiry. A successful workflow, archive existence and original size alone are **PARTIAL R9 evidence**, not proof of its full content. Never silently declare archival completeness.
