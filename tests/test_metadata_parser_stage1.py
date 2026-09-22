from a2s.data.metadata_parser import parse_kern_metadata


def test_metadata_keeps_key_signature_and_tonal_key() -> None:
    metadata = parse_kern_metadata(
        "**kern\n*k[f#c#]\n*D:\n*M3/4\n*MM84\n=1\n4d\n*-\n"
    )
    assert metadata.key_signature == "*k[f#c#]"
    assert metadata.key == "D major"
    assert metadata.meter == "3/4"
    assert metadata.tempo_bpm == 84
