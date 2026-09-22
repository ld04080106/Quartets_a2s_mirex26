from __future__ import annotations

import re


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Exact Levenshtein distance using Myers' bit-vector algorithm.

    Python big integers make this substantially faster than a Python-level
    O(n*m) matrix for dense, jittered score grids while preserving the exact
    insertion/deletion/substitution distance.
    """
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    pattern = reference
    width = len(pattern)
    full_mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    equality: dict[str, int] = {}
    for index, token in enumerate(pattern):
        equality[token] = equality.get(token, 0) | (1 << index)

    positive = full_mask
    negative = 0
    score = width
    for token in hypothesis:
        matches = equality.get(token, 0)
        union = matches | negative
        differences = ((((union & positive) + positive) ^ positive) | union) & full_mask
        horizontal_positive = (negative | ~(differences | positive)) & full_mask
        horizontal_negative = differences & positive
        if horizontal_positive & high_bit:
            score += 1
        elif horizontal_negative & high_bit:
            score -= 1
        shifted = ((horizontal_positive << 1) | 1) & full_mask
        negative = shifted & differences
        positive = ((horizontal_negative << 1) | ~(shifted | differences)) & full_mask
    return score


def kern_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = kern_tokens(reference)
    return edit_distance(ref, kern_tokens(hypothesis)) / max(1, len(ref))


def line_error_rate(reference: str, hypothesis: str) -> float:
    ref = [line for line in reference.replace("\r", "").splitlines() if line.strip()]
    hyp = [line for line in hypothesis.replace("\r", "").splitlines() if line.strip()]
    return edit_distance(ref, hyp) / max(1, len(ref))
