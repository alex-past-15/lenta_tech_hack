"""Склейка наблюдений одного ценника с разных кадров и голосование по полям."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional

from .schema import TagObservation, TagRecord


def norm_name(name: str) -> str:
    return " ".join(name.lower().split())


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, norm_name(a), norm_name(b)).ratio()


def _vote(values: list[tuple[object, float]]) -> Optional[object]:
    score: dict[object, float] = defaultdict(float)
    for v, w in values:
        if v not in (None, "", 0):
            score[v] += w
    return max(score, key=score.get) if score else None


class TagAggregator:
    def __init__(self, min_observations: int = 2, name_similarity_thr: float = 0.85):
        self.min_obs = min_observations
        self.thr = name_similarity_thr
        self.clusters: list[list[TagObservation]] = []

    def _find_cluster(self, obs: TagObservation) -> Optional[list[TagObservation]]:
        for cl in self.clusters:
            ref = cl[0]
            if obs.barcode and ref.barcode:
                if obs.barcode == ref.barcode:
                    return cl
                continue
            if name_similarity(obs.name, ref.name) >= self.thr:
                return cl
        return None

    def add(self, obs: TagObservation) -> None:
        if obs.price is None and not obs.barcode and not obs.name:
            return
        cl = self._find_cluster(obs)
        if cl is None:
            self.clusters.append([obs])
        else:
            cl.append(obs)

    def results(self, video: str = "") -> list[TagRecord]:
        records = []
        for cl in self.clusters:
            if len(cl) < self.min_obs:
                continue
            w = lambda o: max(o.conf, 1e-3)  # noqa: E731
            best_name = max(cl, key=lambda o: (len(o.name) > 0, o.conf)).name
            records.append(
                TagRecord(
                    name=best_name,
                    price=_vote([(o.price, w(o)) for o in cl]),
                    old_price=_vote([(o.old_price, w(o)) for o in cl]),
                    unit_price=_vote([(o.unit_price, w(o)) for o in cl]),
                    unit=_vote([(o.unit, w(o)) for o in cl]) or "",
                    barcode=_vote([(o.barcode, w(o)) for o in cl]) or "",
                    conf=sum(o.conf for o in cl) / len(cl),
                    n_obs=len(cl),
                    first_ts=min(o.ts for o in cl),
                    last_ts=max(o.ts for o in cl),
                    video=video,
                )
            )
        return sorted(records, key=lambda r: r.first_ts)
