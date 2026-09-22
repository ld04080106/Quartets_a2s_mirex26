import unittest

from a2s.data.kern_parser import kern_pitch_to_midi


class KernPitchTest(unittest.TestCase):
    def test_common_kern_pitches(self):
        cases = [
            ("4c", 60, "C4"), ("4d", 62, "D4"), ("4e", 64, "E4"),
            ("4f#", 66, "F#4"), ("4b-", 70, "Bb4"), ("4cc", 72, "C5"),
            ("4C", 48, "C3"), ("4F#", 54, "F#3"), ("4B-", 58, "Bb3"),
        ]
        for token, pitch, name in cases:
            with self.subTest(token=token):
                self.assertEqual(kern_pitch_to_midi(token), (pitch, name))

    def test_articulation_is_ignored(self):
        self.assertEqual(kern_pitch_to_midi("[4cc#L"), (73, "C#5"))


if __name__ == "__main__":
    unittest.main()
