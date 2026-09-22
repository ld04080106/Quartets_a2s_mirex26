from __future__ import annotations

from a2s.data.metadata_parser import ScoreMetadata, quartet_voice_names


SOURCE_VOICES = ("violin_1", "violin_2", "viola", "cello")


def apply_staves_header(text: str, metadata: ScoreMetadata) -> str:
    """Project a four-**kern prediction into the supplied MIREX header layout.

    Staves-informed metadata may interleave auxiliary spines such as ``**dynam``.
    The acoustic pipeline predicts four voices in SOURCE_VOICES order.  This
    function preserves the supplied header verbatim and maps those predictions
    back to the four ``**kern`` columns described by the header.
    """

    if not metadata.header_lines or len(metadata.kern_indices) != 4:
        return text

    lines = [line for line in text.replace("\r", "").splitlines() if line.strip()]
    if not lines:
        return text
    source_columns = lines[0].split("\t")
    if source_columns != ["**kern"] * 4:
        raise ValueError("staves header projection expects a four-spine **kern prediction")

    # Skip the generated interpretation block; retain data, barlines and terminator.
    body_start = next(
        (index for index, line in enumerate(lines[1:], start=1) if not line.startswith(("*", "!"))),
        len(lines),
    )
    body = lines[body_start:]
    exclusive = metadata.header_lines[0].split("\t")
    if len(exclusive) <= max(metadata.kern_indices):
        raise ValueError("metadata header has inconsistent **kern indices")

    target_voices = quartet_voice_names(metadata)
    if sorted(target_voices) != sorted(SOURCE_VOICES):
        raise ValueError(f"metadata does not describe one quartet voice each: {target_voices}")
    source_for_target = [SOURCE_VOICES.index(voice) for voice in target_voices]
    output = list(metadata.header_lines)

    for line in body:
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"prediction body has {len(fields)} columns, expected 4")
        if line.startswith("="):
            target = [fields[0]] * len(exclusive)
        elif line.startswith("*-"):
            target = ["*-"] * len(exclusive)
        else:
            target = ["."] * len(exclusive)
            for local_target, target_column in enumerate(metadata.kern_indices):
                target[target_column] = fields[source_for_target[local_target]]
        output.append("\t".join(target))
    return "\n".join(output) + "\n"
