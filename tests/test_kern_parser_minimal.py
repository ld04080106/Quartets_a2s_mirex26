import tempfile
import unittest
from pathlib import Path

from a2s.data.kern_parser import parse_kern_file


class MinimalKernParserTest(unittest.TestCase):
    def test_minimal_four_spine_kern(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.krn"
            path.write_text(
                "**kern\t**kern\t**kern\t**kern\n"
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\n"
                "*k[]\t*k[]\t*k[]\t*k[]\n"
                "=1\t=1\t=1\t=1\n"
                "4g\t4e\t4c\t4C\n"
                "4a 4cc\t4f\t4d\t4D\n"
                "2b\t2g\t2e\t2E\n"
                "==\t==\t==\t==\n"
                "*-\t*-\t*-\t*-\n",
                encoding="utf-8",
            )
            score = parse_kern_file(path)
        self.assertEqual(score.voices, ["violin_1", "violin_2", "viola", "cello"])
        self.assertEqual(score.meter, "4/4")
        self.assertEqual(score.key, "*k[]")
        self.assertEqual(score.num_measures, 1)
        self.assertEqual({note.voice for note in score.notes}, set(score.voices))
        self.assertEqual(score.chord_token_count, 1)
        chord = [note for note in score.notes if note.onset_beat == 1.0 and note.voice == "violin_1"]
        self.assertEqual(len(chord), 2)
        self.assertEqual({note.duration_beat for note in chord}, {1.0})


if __name__ == "__main__":
    unittest.main()
