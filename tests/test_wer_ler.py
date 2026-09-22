import random

from a2s.evaluation.wer_ler import edit_distance


def _reference_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1, previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def test_bit_vector_distance_matches_reference_dp() -> None:
    rng = random.Random(17)
    alphabet = ["a", "b", "c", ".", "="]
    for _ in range(200):
        left = [rng.choice(alphabet) for _ in range(rng.randrange(20))]
        right = [rng.choice(alphabet) for _ in range(rng.randrange(20))]
        assert edit_distance(left, right) == _reference_distance(left, right)
