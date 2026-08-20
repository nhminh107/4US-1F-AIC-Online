from typing import Literal

from pydantic import BaseModel


class TaskClassification(BaseModel):
    task: Literal["KIS", "VQA", "TRAKE"]


TaskName = Literal["KIS", "VQA", "TRAKE"]


__all__ = [
    "TaskClassification",
    "TaskName",
]
