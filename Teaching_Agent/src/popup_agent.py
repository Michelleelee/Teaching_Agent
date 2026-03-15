# src/popup_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class PopupDecision(BaseModel):
    should_popup: bool = Field(description="Whether to show a check-in popup now.")
    reason: str = Field(description="Short rationale for logging/debug, not shown to student.")
    message: str = Field(description="Popup text shown to the student, keep it short.")
    cta: Literal["understood", "not_sure", "later"] = Field(description="Primary choices to render.")
    recommended_action: Literal["none", "quick_check", "practice_question"] = Field(
        description="What to do if student clicks 'not sure'."
    )
    recommended_topic_id: Optional[str] = Field(default=None, description="Topic to help with.")


@dataclass
class PopupDeps:
    # 你可以把这些依赖扩展成：db/向量检索器/题库接口等
    # 这里最小化：只把你计算好的特征喂给 agent
    pass


def build_popup_agent(model_name: str) -> Agent[PopupDeps, PopupDecision]:
    instructions = (
        "You are an on-slide learning assistant.\n"
        "Goal: decide whether to interrupt the student with a short check-in popup.\n"
        "Rules:\n"
        "1) Default to NOT interrupting unless signals strongly suggest confusion.\n"
        "2) If you choose to popup, keep message neutral and short.\n"
        "3) Do not blame the student. Offer three options: understood / not sure / later.\n"
        "4) Base decision ONLY on the provided signals.\n"
    )
    return Agent(
        model_name,
        deps_type=PopupDeps,
        output_type=PopupDecision,
        instructions=instructions,
    )
