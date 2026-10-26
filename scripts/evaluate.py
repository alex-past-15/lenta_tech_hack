"""Сравнение предсказаний (outputs/*.json) с разметкой (data/annotations/*.json).

Формат разметки: {"video": "x.mp4", "tags": [{"name": ..., "price": ..., "old_price": ..., "barcode": ...}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricetags.config import load_config, resolve  # noqa: E402
from pricetags.metrics import evaluate_tags, micro_average  # noqa: E402

SHOW = ["f1", "precision", "recall", "price_acc_matched", "price_acc_e2e", "full_match_e2e", "duplicate_rate", "name_cer"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pred_dir = resolve(cfg["paths"]["output_dir"])
    ann_dir = resolve(cfg["paths"]["annotations_dir"])

    per_video, speed = {}, []
    for ann_path in sorted(ann_dir.glob("*.json")):
        pred_path = pred_dir / ann_path.name
        if not pred_path.exists():
            print(f"[skip] нет предсказаний для {ann_path.name}")
            continue
        gt = json.loads(ann_path.read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        per_video[ann_path.stem] = evaluate_tags(pred["tags"], gt["tags"])
        st = pred.get("stats", {})
        if st.get("seconds"):
            speed.append(st["frames_used"] / st["seconds"])

    if not per_video:
        sys.exit("Нет пар разметка/предсказание. Положите разметку в data/annotations и запустите scripts/run.py")

    total = micro_average(list(per_video.values()))
    if speed:
        total["frames_per_sec"] = sum(speed) / len(speed)

    print(f"{'video':<24}" + "".join(f"{k:>20}" for k in SHOW))
    for name, m in {**per_video, "TOTAL": total}.items():
        print(f"{name:<24}" + "".join(f"{m[k]:>20.3f}" for k in SHOW))
    if speed:
        print(f"\nСкорость: {total['frames_per_sec']:.2f} кадров/с")

    out = pred_dir / "metrics.json"
    out.write_text(json.dumps({"per_video": per_video, "total": total}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
