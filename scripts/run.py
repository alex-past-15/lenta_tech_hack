"""Прогон пайплайна по видео из data/videos (или по одному файлу)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricetags.config import load_config, resolve  # noqa: E402
from pricetags.pipeline import Pipeline, save_results  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None, help="один файл вместо всей папки data/videos")
    args = ap.parse_args()

    cfg = load_config(args.config)
    videos = [Path(args.video)] if args.video else sorted(
        p for p in resolve(cfg["paths"]["videos_dir"]).iterdir() if p.suffix.lower() in VIDEO_EXT
    )
    if not videos:
        sys.exit("В data/videos нет видео")

    pipe = Pipeline(cfg)
    out_dir = resolve(cfg["paths"]["output_dir"])
    for path in videos:
        records, stats = pipe.process_video(path)
        save_results(records, stats, out_dir)
        print(f"{path.name}: {stats.tags} ценников, {stats.frames_used} кадров, {stats.seconds:.1f} c")


if __name__ == "__main__":
    main()
