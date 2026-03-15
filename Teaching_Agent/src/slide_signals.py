# src/slide_signals.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Optional


class SlideDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    SAME = "same"
    UNKNOWN = "unknown"


@dataclass
class SlideEvent:
    ts: float
    prev_topic: Optional[str]
    curr_topic: str
    dwell_prev_s: float
    direction: SlideDirection
    backtrack_span: int  # how many topics jumped back/forward based on observed order
    toggles_120s: int    # how many direction toggles in last 120s


@dataclass
class SlideTracker:
    """
    仅依赖 topic_id 序列（按 session 首次出现建立顺序），用于方向判定与停留时长。
    """
    topic_order: dict[str, int] = field(default_factory=dict)
    last_topic: Optional[str] = None
    last_ts: float = field(default_factory=lambda: time())

    # for toggle detection
    _dir_hist: list[tuple[float, SlideDirection]] = field(default_factory=list)

    def _order_idx(self, topic: str) -> int:
        if topic not in self.topic_order:
            self.topic_order[topic] = len(self.topic_order)
        return self.topic_order[topic]

    def update(self, curr_topic: str, now: Optional[float] = None) -> Optional[SlideEvent]:
        now = time() if now is None else now

        if self.last_topic is None:
            # first observation: initialize
            self._order_idx(curr_topic)
            self.last_topic = curr_topic
            self.last_ts = now
            return None

        prev = self.last_topic
        dwell_prev = max(0.0, now - self.last_ts)

        prev_idx = self._order_idx(prev)
        curr_idx = self._order_idx(curr_topic)

        if curr_topic == prev:
            direction = SlideDirection.SAME
            span = 0
        elif curr_idx > prev_idx:
            direction = SlideDirection.FORWARD
            span = curr_idx - prev_idx
        elif curr_idx < prev_idx:
            direction = SlideDirection.BACKWARD
            span = prev_idx - curr_idx
        else:
            direction = SlideDirection.UNKNOWN
            span = 0

        # update toggle hist (ignore SAME)
        if direction in (SlideDirection.FORWARD, SlideDirection.BACKWARD):
            self._dir_hist.append((now, direction))
        # keep last 120s
        cutoff = now - 120.0
        self._dir_hist = [(t, d) for (t, d) in self._dir_hist if t >= cutoff]

        toggles = 0
        for i in range(1, len(self._dir_hist)):
            if self._dir_hist[i][1] != self._dir_hist[i - 1][1]:
                toggles += 1

        ev = SlideEvent(
            ts=now,
            prev_topic=prev,
            curr_topic=curr_topic,
            dwell_prev_s=dwell_prev,
            direction=direction,
            backtrack_span=span,
            toggles_120s=toggles,
        )

        # advance state
        self.last_topic = curr_topic
        self.last_ts = now
        return ev
