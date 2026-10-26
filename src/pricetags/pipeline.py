from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from .aggregate import TagAggregator
from .config import resolve
from .schema import RunStats, TagRecord
from .vlm import VLMReader

CSV_FIELDS = [
    "video", "name", "price", "old_price", "unit_price", "unit",
    "barcode", "conf", "n_obs", "first_ts", "last_ts",
]


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        v = cfg["vlm"]
        self.vlm = VLMReader(
            v["base_url"], v["model"], v["api_key_env"], v["max_tokens"],
            v["jpeg_quality"], v["timeout"], v["max_retries"],
        )

    def process_video(self, path: Path) -> tuple[list[TagRecord], RunStats]:
        from .video import iter_frames

        v, a = self.cfg["video"], self.cfg["aggregation"]
        agg = TagAggregator(a["min_observations"], a["name_similarity"])
        stats = RunStats(video=path.name)
        t0 = time.perf_counter()

        for idx, ts, frame in iter_frames(str(path), v["sample_fps"], v["min_sharpness"], v["max_side"]):
            stats.frames_used += 1
            for obs in self.vlm.read(frame):
                obs.frame_idx, obs.ts = idx, ts
                agg.add(obs)
                stats.observations += 1

        records = agg.results(video=path.name)
        stats.tags = len(records)
        stats.seconds = time.perf_counter() - t0
        return records, stats


def save_results(records: list[TagRecord], stats: RunStats, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(stats.video).stem
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"video": stats.video, "tags": [r.to_dict() for r in records], "stats": asdict(stats)},
                  f, ensure_ascii=False, indent=2)
    with open(out_dir / f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_dict() for r in records)
