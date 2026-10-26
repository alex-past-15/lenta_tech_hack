from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

@dataclass
class TagObservation:
    """Один ценник, распознанный на одном кадре."""

    name: str = ""
    price: Optional[float] = None
    old_price: Optional[float] = None
    unit_price: Optional[float] = None
    unit: str = ""
    barcode: str = ""
    conf: float = 0.0
    frame_idx: int = 0
    ts: float = 0.0


@dataclass
class TagRecord:
    """Итоговая запись о ценнике после агрегации по кадрам."""

    name: str
    price: Optional[float]
    old_price: Optional[float]
    unit_price: Optional[float]
    unit: str
    barcode: str
    conf: float
    n_obs: int
    first_ts: float
    last_ts: float
    video: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunStats:
    video: str = ""
    frames_read: int = 0
    frames_used: int = 0
    observations: int = 0
    tags: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)
