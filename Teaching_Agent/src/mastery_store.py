# src/mastery_store.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import time
from typing import Any


@dataclass
class TopicStats:
    n_attempted: int = 0
    n_correct: int = 0
    ema_acc: float = 0.5
    last_answer_ts: float = 0.0


@dataclass
class PopupThrottle:
    last_popup_ts_by_topic: dict[str, float] = field(default_factory=dict)
    prompts_last_hour: list[float] = field(default_factory=list)


@dataclass
class StudentState:
    # 兼容你现有字段：target_difficulty / total_answered 等可以留在 raw 里，不破坏旧逻辑
    topic_stats: dict[str, TopicStats] = field(default_factory=dict)
    popup_throttle: PopupThrottle = field(default_factory=PopupThrottle)

    def to_json(self) -> dict[str, Any]:
        return {
            "topic_stats": {k: asdict(v) for k, v in self.topic_stats.items()},
            "popup_throttle": {
                "last_popup_ts_by_topic": self.popup_throttle.last_popup_ts_by_topic,
                "prompts_last_hour": self.popup_throttle.prompts_last_hour,
            },
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "StudentState":
        ts = {}
        for k, v in (d.get("topic_stats") or {}).items():
            ts[k] = TopicStats(**v)
        pt_raw = d.get("popup_throttle") or {}
        pt = PopupThrottle(
            last_popup_ts_by_topic=pt_raw.get("last_popup_ts_by_topic") or {},
            prompts_last_hour=pt_raw.get("prompts_last_hour") or [],
        )
        return StudentState(topic_stats=ts, popup_throttle=pt)


class MasteryStore:
    """
    用一个 JSON 文件持久化学生状态。你可以把它合并进你现有的 profile 文件里；
    这里给的是“最小侵入”：在 profile 里新增一个 key：student_state。
    """
    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)

    def load_profile_raw(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            return {}
        return json.loads(self.profile_path.read_text(encoding="utf-8"))

    def save_profile_raw(self, raw: dict[str, Any]) -> None:
        self.profile_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_student_state(self) -> StudentState:
        raw = self.load_profile_raw()
        ss = raw.get("student_state") or {}
        return StudentState.from_json(ss)

    def save_student_state(self, ss: StudentState) -> None:
        raw = self.load_profile_raw()
        raw["student_state"] = ss.to_json()
        self.save_profile_raw(raw)

    def update_answer(self, topic_id: str, is_correct: bool, alpha: float = 0.15) -> None:
        ss = self.load_student_state()
        st = ss.topic_stats.get(topic_id, TopicStats())
        st.n_attempted += 1
        if is_correct:
            st.n_correct += 1
        target = 1.0 if is_correct else 0.0
        st.ema_acc = (1.0 - alpha) * st.ema_acc + alpha * target
        st.last_answer_ts = time()
        ss.topic_stats[topic_id] = st
        self.save_student_state(ss)

    def get_topic_snapshot(self, topic_id: str) -> dict[str, Any]:
        ss = self.load_student_state()
        st = ss.topic_stats.get(topic_id, TopicStats())
        acc = (st.n_correct / st.n_attempted) if st.n_attempted > 0 else None
        return {
            "topic_id": topic_id,
            "n_attempted": st.n_attempted,
            "n_correct": st.n_correct,
            "acc": acc,
            "ema_acc": st.ema_acc,
            "last_answer_ts": st.last_answer_ts,
        }
