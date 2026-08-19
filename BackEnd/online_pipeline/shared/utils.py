import re


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


__all__ = ["strip_thinking_blocks"]
