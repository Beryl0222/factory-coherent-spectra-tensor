"""分窗频谱算子的行为核对：公共时格、联合覆盖、版本更正与快照凭据。"""
import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from spectral_op import (
    DELIVERABLE,
    INSUFFICIENT,
    NO_COMMON_ENERGY,
    SpectralOperator,
    estimate_workspace,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def build_operator(records):
    return SpectralOperator(
        nfft=records["nfft"],
        window=records["window"],
        starts=records["starts"],
        rate=records["rate"],
        offset_ticks=records["offset_ticks"],
        response=records["response"],
        bands=records["bands"],
    )


def report(op, arrays, block):
    return op.report_block(
        trigger=block["trigger"],
        channel=block["channel"],
        block_id=block["id"],
        rev=block["rev"],
        start=block["start"],
        samples=arrays[block["tensor"]],
    )


def ref_fold(win, rate, segs):
    """独立参照：segs 为 (左样本, 右样本, [(频点, 权重, 左响应, 右响应)]) 的列表。"""
    win = np.asarray(win, float)
    nfreq = len(win) // 2 + 1
    fold = np.full(nfreq, 2.0)
    fold[0] = 1.0
    if len(win) % 2 == 0:
        fold[-1] = 1.0
    scale = fold / (rate * float(win @ win))
    sxx = syy = 0.0
    sxy = 0j
    denom = 0.0
    for left, right, bins in segs:
        L = np.fft.rfft(np.asarray(left, float) * win)
        R = np.fft.rfft(np.asarray(right, float) * win)
        for f, w, hl, hr in bins:
            cl = L[f] / complex(*hl)
            cr = R[f] / complex(*hr)
            ww = w * scale[f]
            sxx += ww * abs(cl) ** 2
            syy += ww * abs(cr) ** 2
            sxy += ww * cl * np.conj(cr)
            denom += w
    return sxx, syy, sxy, denom


class ScenarioBase(unittest.TestCase):
    """按 frozen_after_block 拆开的标准回放。"""

    def setUp(self):
        self.records, self.arrays = load_case(DATA / "capture.json")
        self.op = build_operator(self.records)
        blocks = self.records["blocks"]
        k = self.records["frozen_after_block"]
        self.pre_receipts = [report(self.op, self.arrays, b) for b in blocks[:k]]
        fr = self.records["frozen_request"]
        self.snap = self.op.claim(fr["trigger"], fr["pair"], fr["band"])
        self.post_receipts = [report(self.op, self.arrays, b) for b in blocks[k:]]
        self.fr = fr

    def ref_segments_ab_low_frozen(self):
        a1, b1, bridge = (self.arrays[k] for k in ("a1", "b1", "b_bridge"))
        win = self.records["window"]
        return [
            (a1[0:8], b1[0:8], [(1, 1.0, (1, 1), (1, 0))]),
            (a1[4:12], np.concatenate([b1[4:8], bridge[2:6]]), [(1, 1.0, (1, 1), (1, 0))]),
        ], win


class GridPlacementTests(ScenarioBase):
    def test_block_grid_and_overlap_dedup(self):
        # 更正后 b 覆盖 [0,20)，三段分别来自 b1、b-bridge、b2r
        tl = self.op._timeline("t0", "b")
        spans = [(s, e) for s, e, _, _ in tl.segments]
        self.assertEqual(spans, [(0, 8), (8, 12), (12, 20)])
        # 冻结期覆盖需回到更正前：另建算子核对
        op = build_operator(self.records)
        for b in self.records["blocks"][: self.records["frozen_after_block"]]:
            report(op, self.arrays, b)
        tl = op._timeline("t0", "b")
        spans = [(s, e) for s, e, _, _ in tl.segments]
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 16)
        for (_, e), (s, _) in zip(spans, spans[1:]):
            self.assertEqual(e, s)  # 段间无缺口：覆盖 [0,16)
        self.assertEqual(tl.conflicts, frozenset())

    def test_window_spanning_response_versions_is_invalid(self):
        # 窗 2(格 [8,16))数据完整，但跨 b 的两套响应有效期，两路配对都不纳入
        v = self.op.query("t0", ("a", "b"), "low")
        self.assertEqual(v.support, ((0, 1), (3, 3)))
        # 窗 2 对单版本响应的 a、c 有效
        vac = self.op.query("t0", ("a", "c"), "low")
        self.assertEqual(vac.support, ((0, 3),))

    def test_triggers_are_separated(self):
        # b-other 属 t1，不进 t0 的联合覆盖；t1 无配对
        v = self.op.query("t1", ("a", "b"), "low")
        self.assertEqual(v.state, INSUFFICIENT)
        self.assertEqual(v.coverage, 0)
        board = self.op.status()
        self.assertEqual(board["triggers"]["t1"], ["b"])
        self.assertFalse(any(h["trigger"] == "t1" for h in board["handoffs"]))


class JointCoverageTests(ScenarioBase):
    def test_frozen_quantities_match_joint_reference(self):
        segs, win = self.ref_segments_ab_low_frozen()
        sxx, syy, sxy, denom = ref_fold(win, self.records["rate"], segs)
        v = self.snap  # 冻结时刻领用的快照
        self.assertEqual(v.coverage, 2)
        self.assertAlmostEqual(v.sxx, sxx, places=12)
        self.assertAlmostEqual(v.syy, syy, places=12)
        self.assertAlmostEqual(v.sxy, sxy, places=12)
        self.assertAlmostEqual(v.denom, denom, places=12)
        self.assertAlmostEqual(v.coherence, abs(sxy) ** 2 / (sxx * syy), places=12)

    def test_denominator_excludes_never_joint_windows(self):
        # 反例：各自平均会把 a 的窗 2、3 纳入分母
        a1 = self.arrays["a1"]
        segs, win = self.ref_segments_ab_low_frozen()
        _, _, sxy, _ = ref_fold(win, self.records["rate"], segs)
        naive_auto_a, _, _, _ = ref_fold(
            win,
            self.records["rate"],
            [(a1[s : s + 8], a1[s : s + 8], [(1, 1.0, (1, 1), (1, 1))]) for s in (0, 4, 8, 12)],
        )
        _, syy, _, _ = ref_fold(win, self.records["rate"], segs)
        naive_coh = abs(sxy) ** 2 / (naive_auto_a * syy)
        v = self.op.query("t0", ("a", "b"), "low")
        self.assertGreater(abs(naive_coh - v.coherence), 1e-3)

    def test_correction_adds_only_joint_windows(self):
        # 更正后 b 覆盖 [0,20)，窗 3(格 [12,20))落入第二套响应，纳入窗 0、1、3
        b2r = self.arrays["b2r"]
        a1 = self.arrays["a1"]
        segs, win = self.ref_segments_ab_low_frozen()
        segs.append((a1[12:20], b2r[0:8], [(1, 1.0, (1, 1), (0, 1)), (2, 2.0, (1, 0), (1, 0))]))
        sxx, syy, sxy, denom = ref_fold(win, self.records["rate"], segs)
        v = self.op.query("t0", ("a", "b"), "low")
        self.assertEqual(v.coverage, 3)
        self.assertAlmostEqual(v.denom, denom, places=12)
        self.assertAlmostEqual(v.sxy, sxy, places=12)
        self.assertAlmostEqual(v.coherence, abs(sxy) ** 2 / (sxx * syy), places=12)

    def test_band_coherence_is_ratio_of_weighted_sums(self):
        # 带内不平均逐频相干：先按频点权重汇聚再取比值
        v = self.op.query("t0", ("a", "b"), "low")
        a1, b1, bridge, b2r = (self.arrays[k] for k in ("a1", "b1", "b_bridge", "b2r"))
        win = self.records["window"]
        bin1 = [
            (a1[0:8], b1[0:8], [(1, 1.0, (1, 1), (1, 0))]),
            (a1[4:12], np.concatenate([b1[4:8], bridge[2:6]]), [(1, 1.0, (1, 1), (1, 0))]),
            (a1[12:20], b2r[0:8], [(1, 1.0, (1, 1), (0, 1))]),
        ]
        sxx1, syy1, sxy1, _ = ref_fold(win, self.records["rate"], bin1)
        sxx2, syy2, sxy2, _ = ref_fold(
            win, self.records["rate"], [(a1[12:20], b2r[0:8], [(2, 1.0, (1, 0), (1, 0))])]
        )
        coh1 = abs(sxy1) ** 2 / (sxx1 * syy1)
        coh2 = abs(sxy2) ** 2 / (sxx2 * syy2)
        mean_of_coh = (1.0 * coh1 + 2.0 * coh2) / 3.0
        ratio_of_sums = abs(sxy1 + 2 * sxy2) ** 2 / ((sxx1 + 2 * sxx2) * (syy1 + 2 * syy2))
        self.assertAlmostEqual(v.coherence, ratio_of_sums, places=12)
        self.assertGreater(abs(mean_of_coh - v.coherence), 1e-3)

    def test_zero_and_unlisted_response_bins_not_counted(self):
        # b 第一套响应在频点 2 为零：harmonic(仅频点 2)只有窗 3 计入
        v = self.op.query("t0", ("a", "b"), "harmonic")
        self.assertEqual(v.coverage, 1)
        self.assertEqual(v.support, ((3, 3),))
        self.assertEqual(v.state, INSUFFICIENT)
        # mid 在窗 0、1 只剩频点 3 可用，分母权重和 = 1+1+2
        vm = self.op.query("t0", ("a", "b"), "mid")
        self.assertEqual(vm.coverage, 3)
        self.assertAlmostEqual(vm.denom, 4.0, places=12)


class VerdictStateTests(ScenarioBase):
    def test_states(self):
        self.assertEqual(self.op.query("t0", ("a", "b"), "low").state, DELIVERABLE)
        self.assertEqual(self.op.query("t0", ("a", "b"), "harmonic").state, INSUFFICIENT)
        # c 全零：覆盖达标但共同能量为零，相干无定义
        v = self.op.query("t0", ("a", "c"), "low")
        self.assertEqual(v.state, NO_COMMON_ENERGY)
        self.assertEqual(v.coverage, 4)
        self.assertIsNone(v.coherence)
        self.assertIsNone(self.op.query("t0", ("b", "c"), "mid").coherence)

    def test_pair_orientation(self):
        ab = self.op.query("t0", ("a", "b"), "low")
        ba = self.op.query("t0", ("b", "a"), "low")
        self.assertAlmostEqual(ba.sxy, ab.sxy.conjugate(), places=15)
        self.assertEqual(ba.auto_left, ab.auto_right)
        self.assertEqual(ba.auto_right, ab.auto_left)
        self.assertAlmostEqual(ba.coherence, ab.coherence, places=15)


class CorrectionAndSnapshotTests(ScenarioBase):
    def test_receipts(self):
        statuses = [r.status for r in self.post_receipts]
        self.assertEqual(statuses, ["ingested", "duplicate"])
        self.assertEqual(self.post_receipts[0].affected_windows, (2, 3))

    def test_snapshot_frozen_against_later_corrections(self):
        frozen = self.op.fetch(self.snap.credential)
        self.assertEqual(frozen, self.snap)
        self.assertEqual(frozen.coverage, 2)
        self.assertEqual(frozen.support, ((0, 1),))
        current = self.op.query("t0", ("a", "b"), "low")
        self.assertEqual(current.coverage, 3)
        self.assertNotEqual(current.sxy, frozen.sxy)

    def test_same_rev_conflict_rejected_and_state_preserved(self):
        before = self.op.query("t0", ("a", "b"), "low")
        r = self.op.report_block(
            trigger="t0", channel="b", block_id="b2", rev=2, start=10,
            samples=np.ones(8),
        )
        self.assertEqual(r.status, "rejected")
        after = self.op.query("t0", ("a", "b"), "low")
        self.assertEqual(before, after)

    def test_stale_rev_ignored(self):
        r = self.op.report_block(
            trigger="t0", channel="b", block_id="b2", rev=1, start=10,
            samples=self.arrays["b2"],
        )
        self.assertEqual(r.status, "stale")

    def test_correction_isolates_other_pairs(self):
        op = build_operator(self.records)
        for b in self.records["blocks"][: self.records["frozen_after_block"]]:
            report(op, self.arrays, b)
        before = op.query("t0", ("a", "c"), "low")
        report(op, self.arrays, self.records["blocks"][6])  # b2 rev2
        after = op.query("t0", ("a", "c"), "low")
        self.assertEqual(before, after)  # 同输入的其余可信对不受版本更正影响

    def test_claimed_snapshot_of_other_pair_untouched(self):
        op = build_operator(self.records)
        for b in self.records["blocks"][: self.records["frozen_after_block"]]:
            report(op, self.arrays, b)
        snap = op.claim("t0", ("a", "c"), "low")
        report(op, self.arrays, self.records["blocks"][6])
        self.assertEqual(op.fetch(snap.credential), snap)


class ConflictResolutionTests(unittest.TestCase):
    def make_operator(self):
        return SpectralOperator(
            nfft=4,
            window=[1.0, 1.0, 1.0, 1.0],
            starts=[0],
            rate=8,
            offset_ticks={"x": 0, "y": 0},
            response={
                "x": [{"effective": [0, 100], "re_im": [[1, 0], [1, 0], [1, 0]]}],
                "y": [{"effective": [0, 100], "re_im": [[1, 0], [1, 0], [1, 0]]}],
            },
            bands=[{"id": "only", "weights": {"1": 1}, "min_windows": 1}],
        )

    def test_conflicting_blocks_mark_cells_unusable_then_resolve(self):
        op = self.make_operator()
        op.report_block(trigger="t", channel="x", block_id="p", rev=1, start=0,
                        samples=[1.0, 2.0, 3.0, 4.0])
        op.report_block(trigger="t", channel="y", block_id="q", rev=1, start=0,
                        samples=[4.0, 3.0, 2.0, 1.0])
        self.assertEqual(op.query("t", ("x", "y"), "only").coverage, 1)
        # 不同块同时格读数不一致：时格不可用，窗口掉出联合覆盖
        op.report_block(trigger="t", channel="x", block_id="r", rev=1, start=0,
                        samples=[9.0, 9.0, 9.0, 9.0])
        self.assertEqual(op.query("t", ("x", "y"), "only").coverage, 0)
        # 更高 rev 与既有块一致后，冲突消解，窗口恢复
        op.report_block(trigger="t", channel="x", block_id="r", rev=2, start=0,
                        samples=[1.0, 2.0, 3.0, 4.0])
        v = op.query("t", ("x", "y"), "only")
        self.assertEqual(v.coverage, 1)
        self.assertEqual(v.state, DELIVERABLE)

    def test_second_stream_attaches_with_own_handoffs(self):
        op = self.make_operator()
        op.report_block(trigger="t1", channel="x", block_id="u1", rev=1, start=0,
                        samples=[1.0, 2.0, 3.0, 4.0])
        board = op.status()
        self.assertEqual(board["triggers"], {"t1": ["x"]})
        self.assertEqual(board["handoffs"], [])
        op.report_block(trigger="t1", channel="y", block_id="v1", rev=1, start=0,
                        samples=[4.0, 3.0, 2.0, 1.0])
        board = op.status()
        self.assertEqual(board["triggers"], {"t1": ["x", "y"]})
        self.assertEqual(len(board["handoffs"]), 1)
        h = board["handoffs"][0]
        self.assertEqual((h["trigger"], h["pair"], h["band"]), ("t1", ["x", "y"], "only"))
        self.assertEqual(h["state"], DELIVERABLE)
        self.assertEqual(h["support"], [{"windows": [0, 0], "grid": [0, 4]}])


class CapacityTests(unittest.TestCase):
    def test_workspace_estimate_within_capacity(self):
        cap = json.loads((DATA / "capacity.json").read_text(encoding="utf-8"))
        windows = (cap["samples_per_channel"] - cap["nfft"]) // cap["hop"] + 1
        estimate = estimate_workspace(
            channels=cap["channels"],
            windows_per_trigger=windows,
            bands=cap["bands"],
        )
        self.assertLess(estimate, cap["max_workspace_bytes"])

    def test_operator_workspace_small_on_synthetic_case(self):
        records, arrays = load_case(DATA / "capture.json")
        op = build_operator(records)
        for b in records["blocks"]:
            report(op, arrays, b)
        op.claim("t0", ("a", "b"), "low")
        self.assertLess(op.workspace_bytes()["total"], 1 << 20)


if __name__ == "__main__":
    unittest.main()
