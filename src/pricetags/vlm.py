"""Чтение ценников с помощью VLM через OpenAI-совместимый API (vLLM, Ollama, OpenAI и т. п.)."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Optional

from .schema import TagObservation

PROMPT = """Ты видишь изображение из торгового зала магазина. Найди на нём все ценники и верни JSON-массив.
Для каждого ценника объект с полями:
  "name": название товара как напечатано на ценнике, строка;
  "price": цена, которую платит покупатель (крупная), число с точкой;
  "old_price": старая цена до скидки, если она есть, иначе null;
  "unit_price": цена за единицу измерения (кг, л, шт), если есть, иначе null;
  "unit": единица измерения для unit_price ("кг", "л", "шт"), если есть, иначе "";
  "barcode": штрихкод или артикул цифрами, если виден, иначе "".
Не выдумывай: если поле не читается, ставь null или "". Если ценников нет, верни [].
Ответ — только JSON, без пояснений."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    m = re.search(r"\d+(?:[.,]\d+)?", str(value).replace(" ", ""))
    return round(float(m.group(0).replace(",", ".")), 2) if m else None


def parse_response(text: str) -> list[TagObservation]:
    """Достаёт ценники из ответа модели. Битый ответ даёт пустой список."""
    fence = _FENCE_RE.search(text)
    body = fence.group(1) if fence else text
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return []

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        obs = TagObservation(
            name=re.sub(r"\s+", " ", str(it.get("name") or "")).strip(),
            price=_to_float(it.get("price")),
            old_price=_to_float(it.get("old_price")),
            unit_price=_to_float(it.get("unit_price")),
            unit=str(it.get("unit") or "").strip().lower(),
            barcode=re.sub(r"\D", "", str(it.get("barcode") or "")),
            conf=1.0,  # VLM не отдаёт уверенность: все наблюдения голосуют с равным весом
        )
        if obs.price is not None or obs.name:
            out.append(obs)
    return out


class VLMReader:
    def __init__(self, base_url: str, model: str, api_key_env: str, max_tokens: int,
                 jpeg_quality: int, timeout: float, max_retries: int):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "EMPTY"),  # локальному серверу ключ не нужен
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model, self.max_tokens, self.jpeg_quality = model, max_tokens, jpeg_quality

    def _data_url(self, image) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ValueError("Не удалось закодировать изображение")
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def read(self, image) -> list[TagObservation]:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url(image)}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        return parse_response(resp.choices[0].message.content or "")
