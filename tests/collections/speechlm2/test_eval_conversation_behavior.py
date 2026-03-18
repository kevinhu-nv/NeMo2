# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock


def _load_eval_script():
    """Load eval_conversation_behavior.py while mocking its heavy dependencies."""
    mocks = {
        'torchaudio': MagicMock(),
        'torch': MagicMock(),
        'tqdm': MagicMock(),
        'tqdm.tqdm': MagicMock(),
        'nemo': MagicMock(),
        'nemo.collections': MagicMock(),
        'nemo.collections.asr': MagicMock(),
        'openai': MagicMock(),
        'jiwer': MagicMock(),
        'whisper_normalizer': MagicMock(),
        'whisper_normalizer.english': MagicMock(),
    }
    # Ensure submodule mocks are reachable via attribute access on parents
    mocks['whisper_normalizer'].english = mocks['whisper_normalizer.english']

    originals = {}
    for name, mock in mocks.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mock

    try:
        spec = importlib.util.spec_from_file_location(
            "eval_conversation_behavior",
            "scripts/speech_eval/eval_conversation_behavior.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


_module = _load_eval_script()
compute_user_eou_metrics = _module.compute_user_eou_metrics


class TestStreamingASRDelayCorrection(unittest.TestCase):
    """Tests that streaming ASR delay correction makes EOU matching work correctly."""

    DELAY = 1.12  # default streaming_asr_delay

    def _make_seg(self, start, end, text="hello"):
        return {'start': start, 'end': end, 'text': text}

    def _apply_delay_correction(self, segments, delay):
        """Mirrors the correction applied in the script before compute_user_eou_metrics is called."""
        return [{'start': s['start'] - delay, 'end': s['end'] - delay, 'text': s['text']} for s in segments]

    def test_without_delay_correction_segments_do_not_match(self):
        """Without delay correction, predicted timestamps are ~1.12s late and fall outside the tight match window."""
        gt_segments = [self._make_seg(5.0, 8.0)]
        # Predicted timestamps are 1.12s late (as streaming ASR would emit them)
        pred_segments = [self._make_seg(5.0 + self.DELAY, 8.0 + self.DELAY)]

        # Use a tight threshold (0.5s) that does not cover the 1.12s delay
        metrics = compute_user_eou_metrics(pred_segments, gt_segments, match_threshold_sec=0.5)

        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['false_negatives'], 1)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)

    def test_with_delay_correction_segments_match(self):
        """After subtracting the ASR delay, predicted timestamps align with GT and match correctly."""
        gt_segments = [self._make_seg(5.0, 8.0)]
        # Predicted timestamps are 1.12s late — same as above
        pred_segments_raw = [self._make_seg(5.0 + self.DELAY, 8.0 + self.DELAY)]

        # Apply the delay correction (as the script does before calling compute_user_eou_metrics)
        pred_segments_corrected = self._apply_delay_correction(pred_segments_raw, self.DELAY)

        metrics = compute_user_eou_metrics(pred_segments_corrected, gt_segments, match_threshold_sec=0.5)

        self.assertEqual(metrics['true_positives'], 1)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)

    def test_eou_latency_is_zero_after_perfect_correction(self):
        """When predicted timestamps exactly match GT after delay subtraction, EOU latency should be ~0."""
        gt_segments = [self._make_seg(5.0, 8.0)]
        pred_segments_raw = [self._make_seg(5.0 + self.DELAY, 8.0 + self.DELAY)]
        pred_segments_corrected = self._apply_delay_correction(pred_segments_raw, self.DELAY)

        metrics = compute_user_eou_metrics(pred_segments_corrected, gt_segments, match_threshold_sec=0.5)

        self.assertAlmostEqual(metrics['avg_eou_latency'], 0.0, places=5)

    def test_multiple_segments_all_matched_after_correction(self):
        """All segments across a conversation are correctly matched after delay correction."""
        gt_segments = [
            self._make_seg(2.0, 5.0),
            self._make_seg(10.0, 14.0),
            self._make_seg(20.0, 23.5),
        ]
        pred_segments_raw = [self._make_seg(s['start'] + self.DELAY, s['end'] + self.DELAY) for s in gt_segments]
        pred_segments_corrected = self._apply_delay_correction(pred_segments_raw, self.DELAY)

        metrics = compute_user_eou_metrics(pred_segments_corrected, gt_segments, match_threshold_sec=0.5)

        self.assertEqual(metrics['true_positives'], 3)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)

    def test_zero_delay_leaves_timestamps_unchanged(self):
        """A streaming_asr_delay of 0.0 should be a no-op."""
        gt_segments = [self._make_seg(5.0, 8.0)]
        pred_segments = [self._make_seg(5.0, 8.0)]
        corrected = self._apply_delay_correction(pred_segments, delay=0.0)

        self.assertEqual(corrected[0]['start'], 5.0)
        self.assertEqual(corrected[0]['end'], 8.0)

        metrics = compute_user_eou_metrics(corrected, gt_segments, match_threshold_sec=0.5)
        self.assertEqual(metrics['true_positives'], 1)


if __name__ == '__main__':
    unittest.main()
