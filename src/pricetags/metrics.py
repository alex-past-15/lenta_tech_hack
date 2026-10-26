"""Метрики качества. Чистый Python, без cv2 и моделей."""

from __future__ import annotations

from statistics import mean
from typing import Optional

from .aggregate import name_similarity, norm_name

PRICE_TOL = 0.005


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------- чтение полей ----------

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    """Character Error Rate по нормализованным строкам."""
    pred, gt = norm_name(pred), norm_name(gt)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


# ---------- сквозные метрики по уникальным ценникам ----------

def _price_eq(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= PRICE_TOL


def _tag_score(p: dict, g: dict) -> float:
    if p.get("barcode") and g.get("barcode"):
        return 2.0 if p["barcode"] == g["barcode"] else 0.0
    return name_similarity(p.get("name", ""), g.get("name", ""))


def match_tags(preds: list[dict], gts: list[dict], min_sim: float = 0.6) -> list[tuple[int, int]]:
    """Жадно сопоставляет предсказанные и эталонные ценники: сперва по штрихкоду, затем по названию."""
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            score = _tag_score(p, g)
            if score >= min_sim:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j), out.append((i, j))
    return out


def evaluate_tags(preds: list[dict], gts: list[dict]) -> dict[str, float]:
    matches = match_tags(preds, gts)
    tp = len(matches)
    res = _prf(tp, len(preds) - tp, len(gts) - tp)

    price_ok = old_ok = bc_ok = full_ok = 0
    cers = []
    for i, j in matches:
        p, g = preds[i], gts[j]
        pr = _price_eq(p.get("price"), g.get("price"))
        old = _price_eq(p.get("old_price"), g.get("old_price"))
        bc = (p.get("barcode") or "") == (g.get("barcode") or "")
        price_ok += pr
        old_ok += old
        bc_ok += bc
        full_ok += pr and old and bc
        cers.append(cer(p.get("name", ""), g.get("name", "")))

    n_gt = len(gts)
    matched_p = {i for i, _ in matches}
    # дубль: лишнее предсказание, которое похоже на какой-то эталонный ценник
    dups = sum(
        1 for i, p in enumerate(preds)
        if i not in matched_p and any(_tag_score(p, g) >= 0.6 for g in gts)
    )
    res.update(
        {
            "duplicates": dups,
            "duplicate_rate": dups / n_gt if n_gt else 0.0,
            "price_acc_matched": price_ok / tp if tp else 0.0,
            "old_price_acc_matched": old_ok / tp if tp else 0.0,
            "barcode_acc_matched": bc_ok / tp if tp else 0.0,
            "name_cer": mean(cers) if cers else 1.0,
            # строгие метрики: промах по детекции тоже считается ошибкой
            "price_acc_e2e": price_ok / n_gt if n_gt else 0.0,
            "full_match_e2e": full_ok / n_gt if n_gt else 0.0,
            "n_gt": n_gt,
            "n_pred": len(preds),
        }
    )
    return res


def micro_average(per_video: list[dict[str, float]]) -> dict[str, float]:
    """Пересчитывает метрики по сумме TP/FP/FN и усредняет остальное с весом числа эталонных ценников."""
    tp = sum(m["tp"] for m in per_video)
    fp = sum(m["fp"] for m in per_video)
    fn = sum(m["fn"] for m in per_video)
    out = _prf(tp, fp, fn)
    total_gt = sum(m["n_gt"] for m in per_video) or 1
    out["duplicates"] = sum(m["duplicates"] for m in per_video)
    for k in ("price_acc_e2e", "full_match_e2e", "duplicate_rate"):
        out[k] = sum(m[k] * m["n_gt"] for m in per_video) / total_gt
    total_tp = tp or 1
    for k in ("price_acc_matched", "old_price_acc_matched", "barcode_acc_matched", "name_cer"):
        out[k] = sum(m[k] * m["tp"] for m in per_video) / total_tp
    out["n_gt"], out["n_pred"] = total_gt if per_video else 0, sum(m["n_pred"] for m in per_video)
    return out
