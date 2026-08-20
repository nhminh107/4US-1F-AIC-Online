from typing import Literal

from pydantic import BaseModel


class TaskClassification(BaseModel):
    task: Literal["KIS", "VQA"]


TaskName = Literal["KIS", "VQA"]


__all__ = [
    "TaskClassification",
    "TaskName",
]
