from fractions import Fraction

from a2s.evaluation.kern_validity import validate_kern_text
from a2s.stage2_score.kern_writer import write_kern
from a2s.stage2_score.score_grid import Segment


def test_writer_produces_valid_quartet():
    spines = [[Segment(Fraction(0), Fraction(4), (60 + i,))] for i in range(4)]
    text = write_kern(spines)
    result = validate_kern_text(text)
    assert result.valid, result.issues
    assert result.spine_count == 4
