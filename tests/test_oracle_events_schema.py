import unittest

from a2s.data.kern_parser import KernNote, KernScore
from a2s.data.oracle_events import REQUIRED_NOTE_FIELDS, kern_score_to_oracle, validate_oracle_payload


class OracleSchemaTest(unittest.TestCase):
    def test_oracle_event_schema_has_required_fields(self):
        score = KernScore(
            sample_id="sample", spines=["**kern"] * 4,
            voices=["violin_1", "violin_2", "viola", "cello"],
            meter="4/4", key="*k[]", num_measures=1,
            notes=[KernNote(
                voice="violin_1", pitch=74, pitch_name="D5", onset_beat=0.0,
                duration_beat=1.0, offset_beat=1.0, measure_index=1,
                beat_in_measure=0.0, token="4dd",
            )],
        )
        payload = kern_score_to_oracle(score, "Haydn")
        self.assertEqual(payload["sample_id"], "sample")
        self.assertIsInstance(payload["notes"], list)
        self.assertLessEqual(REQUIRED_NOTE_FIELDS, set(payload["notes"][0]))
        self.assertEqual(validate_oracle_payload(payload), [])


if __name__ == "__main__":
    unittest.main()
