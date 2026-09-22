import unittest

from a2s.data.kern_parser import kern_duration_to_beats


class KernDurationTest(unittest.TestCase):
    def test_common_kern_durations(self):
        cases = [
            ("4c", 1.0), ("8c", 0.5), ("16c", 0.25), ("2c", 2.0),
            ("4.c", 1.5), ("4..c", 1.75), ("8r", 0.5),
        ]
        for token, expected in cases:
            with self.subTest(token=token):
                self.assertEqual(kern_duration_to_beats(token), expected)


if __name__ == "__main__":
    unittest.main()
