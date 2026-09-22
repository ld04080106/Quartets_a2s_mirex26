from fractions import Fraction

from a2s.data.metadata_parser import parse_kern_metadata
from a2s.evaluation.kern_validity import validate_kern_text
from a2s.stage2_score.kern_writer import write_kern
from a2s.stage2_score.score_grid import Segment
from a2s.stage2_score.staves_header import apply_staves_header


def test_staves_header_preserves_auxiliary_spines_and_reorders_voices() -> None:
    header = "\n".join((
        "**kern\t**dynam\t**kern\t**dynam\t**kern\t**dynam\t**kern\t**dynam",
        "*Icello\t*Icello\t*Iviola\t*Iviola\t*Ivioln\t*Ivioln\t*Iflt\t*Iflt",
        "*clefF4\t*\t*clefC3\t*\t*clefG2\t*\t*clefG2\t*",
        "*k[f#c#]\t*\t*k[f#c#]\t*\t*k[f#c#]\t*\t*k[f#c#]\t*",
        "*D:\t*\t*D:\t*\t*D:\t*\t*D:\t*",
        "*M4/4\t*\t*M4/4\t*\t*M4/4\t*\t*M4/4\t*",
        "*MM100\t*\t*MM100\t*\t*MM100\t*\t*MM100\t*",
    ))
    metadata = parse_kern_metadata(header)
    # Generated order is violin I, violin II, viola, cello.
    spines = [[Segment(Fraction(0), Fraction(4), (pitch,))] for pitch in (84, 79, 67, 48)]
    result = apply_staves_header(write_kern(spines, tempo_bpm=100), metadata)

    lines = result.splitlines()
    assert lines[:7] == header.splitlines()
    first_data = next(line.split("\t") for line in lines[7:] if not line.startswith(("*", "=")))
    assert first_data[0].endswith("C")       # cello
    assert first_data[2].endswith("g")       # viola
    assert first_data[4].endswith("gg")      # violin II
    assert first_data[6].endswith("ccc")     # violin I
    assert first_data[1::2] == ["."] * 4
    validation = validate_kern_text(result)
    assert validation.valid, validation.issues
    assert validation.spine_count == 8
