# src/popup_policy.py
from __future__ import annotations

from time import time
from typing import Any, Optional

from .mastery_store import MasteryStore
from .slide_signals import SlideEvent, SlideDirection


def compute_confusion_features(
    ev: SlideEvent,
    mastery_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    返回一组结构化特征，直接喂给 PydanticAI agent（不要让它猜）。
    """
    ema = float(mastery_snapshot.get("ema_acc", 0.5))
    n_attempted = int(mastery_snapshot.get("n_attempted", 0))

    # 基础启发式分数（只是作为 feature，而不是最终裁决）
    score = 0.0
    if ev.direction == SlideDirection.BACKWARD:
        score += 0.45
        score += min(0.25, 0.05 * ev.backtrack_span)
        if ev.toggles_120s >= 2:
            score += 0.15
    # 回退后停留（这里 dwell_prev_s 是“上一 topic 的停留”，
    # 若你希望用“回退后在目标 topic 的停留”，需要在 tracker 外层再记录）
    # 作为最小可用，我们用“刚刚在 prev_topic 停留太短就回退”来推断没跟上：
    if ev.direction == SlideDirection.BACKWARD and ev.dwell_prev_s < 60:
        score += 0.15

    # 掌握度低
    if n_attempted >= 2 and ema < 0.6:
        score += 0.25
    elif n_attempted == 0 and ev.direction == SlideDirection.BACKWARD:
        # 从没练过又回退，也算弱信号
        score += 0.10

    score = max(0.0, min(1.0, score))

    return {
        "direction": ev.direction.value,
        "backtrack_span": ev.backtrack_span,
        "toggles_120s": ev.toggles_120s,
        "dwell_prev_s": round(ev.dwell_prev_s, 2),
        "topic_ema_acc": round(ema, 3),
        "topic_n_attempted": n_attempted,
        "heuristic_confusion_score": round(score, 3),
    }


def throttle_allows(
    store: MasteryStore,
    topic_id: str,
    cooldown_s: int = 8 * 60,
    max_per_hour: int = 2,
) -> tuple[bool, Optional[str]]:
    """
    True 表示允许触发；False 表示被限流拦住（返回原因字符串用于日志）。
    """
    ss = store.load_student_state()
    now = time()

    # cooldown by topic
    last = ss.popup_throttle.last_popup_ts_by_topic.get(topic_id, 0.0)
    if now - last < cooldown_s:
        return False, f"cooldown_active (remaining={int(cooldown_s - (now - last))}s)"

    # max per hour
    cutoff = now - 3600.0
    ss.popup_throttle.prompts_last_hour = [t for t in ss.popup_throttle.prompts_last_hour if t >= cutoff]
    if len(ss.popup_throttle.prompts_last_hour) >= max_per_hour:
        return False, "max_per_hour_reached"

    return True, None


def throttle_commit(store: MasteryStore, topic_id: str) -> None:
    ss = store.load_student_state()
    now = time()
    ss.popup_throttle.last_popup_ts_by_topic[topic_id] = now
    ss.popup_throttle.prompts_last_hour.append(now)
    store.save_student_state(ss)
