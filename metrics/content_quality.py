"""Content quality metrics: vague words, repeated sentences."""

import re
from collections import Counter
from parser import LogSession
from .base import BaseMetric

VAGUE_WORDS = [
    "可能", "大概", "或许", "建议", "疑似", "也许", "似乎", "好像",
    "应该", "不确定", "不确定是否", "无法确定", "未必", "不一定",
]

# Split text into sentences by Chinese/English sentence-end punctuation
_SENTENCE_SPLIT_RE = re.compile(r"[。！？.!?\n]+")


class ContentQualityMetric(BaseMetric):
    name = "内容质量"
    category = "content"

    def __init__(self, vague_words: list[str] | None = None):
        self.vague_words = vague_words or VAGUE_WORDS
        # Build regex that matches any of the vague words
        self._vague_re = re.compile(
            "|".join(re.escape(w) for w in sorted(self.vague_words, key=len, reverse=True))
        )

    def compute(self, session: LogSession) -> dict:
        vague_hits = 0
        all_sentences: list[str] = []

        for msg in session.messages:
            for block in msg.content:
                if block.type == "text" and block.text:
                    vague_hits += len(self._vague_re.findall(block.text))
                    parts = _SENTENCE_SPLIT_RE.split(block.text)
                    for p in parts:
                        stripped = p.strip()
                        if len(stripped) >= 6:
                            all_sentences.append(stripped)

        # Count sentences appearing >= 3 times
        sentence_counts = Counter(all_sentences)
        repeat_3plus = sum(1 for c in sentence_counts.values() if c >= 3)

        return {
            "vague_word_hits": vague_hits,
            "sentence_repeat_3plus": repeat_3plus,
        }
