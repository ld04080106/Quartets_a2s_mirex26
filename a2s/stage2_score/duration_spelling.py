from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import re


def token_duration(token: str) -> Fraction:
    match = re.match(r"^[\[(_&]*?(\d+)(\.*)", token)
    if not match:
        raise ValueError(f"invalid duration token: {token}")
    reciprocal, dot_text = match.groups()
    base = Fraction(4, int(reciprocal))
    result, part = base, base
    for _ in dot_text:
        part /= 2
        result += part
    return result


def _candidate_table() -> list[tuple[Fraction, str]]:
    # Some reciprocals denote the same duration (16 and 24. are both 1/4
    # beat). Prefer undotted binary values so output resembles normal notation.
    best: dict[Fraction, tuple[tuple[int, int, int], str]] = {}
    for denom in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192):
        for dots in (0, 1, 2):
            token = f"{denom}{'.' * dots}"
            value = token_duration(token)
            is_binary = denom > 0 and denom & (denom - 1) == 0
            rank = (dots, 0 if is_binary else 1, denom)
            if value not in best or rank < best[value][0]:
                best[value] = (rank, token)
    return sorted(((value, item[1]) for value, item in best.items()), reverse=True)


_CANDIDATES = _candidate_table()


@lru_cache(maxsize=4096)
def _spell_ticks(target: int) -> tuple[str, ...] | None:
    candidates = [(int(value * 192), token) for value, token in _CANDIDATES]
    # A largest-first greedy pass can get trapped (5/48 beats: 64. leaves an
    # unspellable 1/96). Dynamic programming finds an exact, minimum-token sum;
    # candidate order retains the conventional spelling preference above.
    best: list[list[str] | None] = [None] * (target + 1)
    best[0] = []
    for amount in range(1, target + 1):
        for ticks, token in candidates:
            if ticks <= amount and best[amount - ticks] is not None:
                proposal = best[amount - ticks] + [token]
                if best[amount] is None or len(proposal) < len(best[amount]):
                    best[amount] = proposal
    if best[target] is None:
        return None
    return tuple(reversed(best[target]))


def spell_duration(duration_beat: float | Fraction) -> list[str]:
    remaining = Fraction(duration_beat).limit_denominator(192)
    if remaining <= 0:
        raise ValueError("duration must be positive")
    tick_value = remaining * 192
    if tick_value.denominator != 1:
        raise ValueError(f"duration {duration_beat} is outside the 1/192-beat grid")
    result = _spell_ticks(int(tick_value))
    if result is None:
        raise ValueError(f"duration {duration_beat} is not spellable")
    return list(result)
