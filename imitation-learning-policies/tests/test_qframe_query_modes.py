import torch
import unittest


class QFrameQueryModesTest(unittest.TestCase):
    def test_fusion_selects_high_and_low_rows(self):
        from imitation_learning.utils.qframe_query_modes import (
            assign_qframe_rows,
            fuse_query_scores,
        )

        image_scores = torch.tensor([[0.10, 0.40, 0.30, 0.90, 0.80]])
        text_scores = torch.tensor([[0.95, 0.20, 0.60, 0.10, 0.30]])

        fused = fuse_query_scores(image_scores, text_scores, alpha=0.5)
        self.assertTrue(
            torch.allclose(
                fused,
                torch.tensor([[0.525, 0.300, 0.450, 0.500, 0.550]]),
            )
        )

        rows = assign_qframe_rows(
            fused[0],
            max_candidates=5,
            high_topk=2,
            low_topk=2,
        )
        by_index = {row["history_index"]: row for row in rows}

        self.assertEqual(by_index[4]["kind"], "high")
        self.assertEqual(by_index[0]["kind"], "high")
        self.assertEqual(by_index[3]["kind"], "low")
        self.assertEqual(by_index[2]["kind"], "low")
        self.assertEqual(by_index[1]["kind"], "candidate")


    def test_candidate_indices_are_evenly_spaced(self):
        from imitation_learning.utils.qframe_query_modes import qframe_candidate_indices

        self.assertEqual(
            qframe_candidate_indices(history_len=5, max_candidates=8),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            qframe_candidate_indices(history_len=9, max_candidates=4),
            [0, 2, 5, 8],
        )
        self.assertEqual(qframe_candidate_indices(history_len=9, max_candidates=1), [0])


if __name__ == "__main__":
    unittest.main()
