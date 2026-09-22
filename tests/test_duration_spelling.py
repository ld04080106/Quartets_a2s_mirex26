from fractions import Fraction

from a2s.stage2_score.duration_spelling import spell_duration, token_duration


def test_duration_spelling_is_exact():
    tokens = spell_duration(Fraction(5, 4))
    assert sum((token_duration(token) for token in tokens), Fraction()) == Fraction(5, 4)


def test_prefers_conventional_binary_spelling_for_equivalent_value():
    assert spell_duration(Fraction(1, 4)) == ["16"]
    assert spell_duration(Fraction(1, 2)) == ["8"]


def test_non_greedy_duration_is_still_exact():
    tokens = spell_duration(Fraction(5, 48))
    assert sum((token_duration(token) for token in tokens), Fraction()) == Fraction(5, 48)
