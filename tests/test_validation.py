from a2s.evaluation.kern_validity import validate_kern_text


def test_validation_rejects_unequal_columns():
    result = validate_kern_text("**kern\t**kern\n1c\n*-\t*-\n")
    assert not result.valid
