"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(len(r["window"]), r["nfft"])
        self.assertIn([0, 0], r["response"]["b"][0]["re_im"])
        self.assertGreater(a["b2r"].size, a["b2"].size)
        self.assertFalse(np.any(a["c1"]))
        by_tensor = {block["tensor"]: block for block in r["blocks"]}
        self.assertEqual(by_tensor["b1"]["trigger"], by_tensor["b_bridge"]["trigger"])
        overlap = by_tensor["b_bridge"]["start"] - by_tensor["b1"]["start"]
        self.assertGreater(a["b1"].size, overlap)
        np.testing.assert_array_equal(a["b1"][overlap:], a["b_bridge"][: a["b1"].size - overlap])
        prefix = r["blocks"][:r["frozen_after_block"]]
        self.assertIn(by_tensor["b_other"], prefix)
        self.assertNotEqual(by_tensor["b_other"]["trigger"], r["frozen_request"]["trigger"])
        self.assertNotIn(by_tensor["b2r"], prefix)
