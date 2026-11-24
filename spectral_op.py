"""分窗频谱算子：异步采集的相干谱张量交付。

采集清单给出触发时钟(offset_ticks)、块起点与采样率，算子把各通道数据块
放到公共样本格，在公共窗口上按"两路共同纳入"的口径累计自功率、互谱与
幅值平方相干，并按频段权重汇聚成交付特征。

约定：
- 数据块以 (trigger, channel, id) 区分，较高 rev 是对同一块的完整替代；
  同块同版重发须内容一致，冲突的上报被拒绝并保留已采入内容。
- 同一触发和通道的不同块在同一时格读数一致只占一格，不一致的时格不可用；
  不同触发的读数各自计算。
- 窗口对某通道可用，当且仅当整个窗口在相同触发下有完整有效读数，且窗口
  落在同一套响应有效期内(跨两套响应的窗口无效)；频点可用还要求该点响应
  可校正(响应为零或未列出即不可校正)。
- 每个 (触发, 配对, 频段) 的相干分母只含两路共同纳入的窗口；频段量先按
  频点权重汇聚同一组联合覆盖窗口的自功率与互谱再取比值，不平均逐频相干；
  分母为零时相干无定义。
- 已领用的谱快照按凭据重查；之后的更正只进入仍开放的结果，不改旧快照的
  谱量与支持窗。
- 工作区只保留 (配对, 频段) 累计器与窗口身份位图，不保留全量复数谱；
  调用方持有的只读输入不计入工作区。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# 交接状态
DELIVERABLE = "deliverable"        # 可消费
INSUFFICIENT = "insufficient"      # 证据还不齐
NO_COMMON_ENERGY = "no_common_energy"  # 未观察到共能量


class _BitSet:
    """窗口身份的紧凑位图，工作区不随窗口数线性膨胀。"""

    __slots__ = ("_bits",)

    def __init__(self, size: int, bits: Optional[bytearray] = None) -> None:
        self._bits = bits if bits is not None else bytearray((size + 7) // 8)

    def add(self, index: int) -> None:
        self._bits[index >> 3] |= 1 << (index & 7)

    def discard(self, index: int) -> None:
        self._bits[index >> 3] &= 0xFF ^ (1 << (index & 7))

    def __contains__(self, index: int) -> bool:
        return bool(self._bits[index >> 3] & (1 << (index & 7)))

    def __eq__(self, other) -> bool:
        return isinstance(other, _BitSet) and self._bits == other._bits

    def copy(self) -> "_BitSet":
        return _BitSet(0, bytearray(self._bits))

    def indices(self) -> Iterator[int]:
        for pos, byte in enumerate(self._bits):
            b = byte
            while b:
                low = b & (-b)
                yield (pos << 3) | (low.bit_length() - 1)
                b ^= low

    def ranges(self) -> Tuple[Tuple[int, int], ...]:
        """连续窗口合并为 (首, 末) 闭区间，便于交接展示。"""
        runs: List[Tuple[int, int]] = []
        for i in self.indices():
            if runs and i == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((i, i))
        return tuple(runs)


@dataclass(frozen=True)
class BandSpec:
    """频段口径：正权频点与按联合覆盖数生效的交付阈值。"""

    band_id: str
    weights: Tuple[Tuple[int, float], ...]  # (频点, 权重)，只保留正权，按频点升序
    min_windows: int

    @classmethod
    def from_record(cls, rec: dict) -> "BandSpec":
        weights = tuple(
            sorted(
                (int(k), float(v))
                for k, v in rec["weights"].items()
                if float(v) > 0
            )
        )
        return cls(str(rec["id"]), weights, int(rec["min_windows"]))


class ResponseBook:
    """单通道的响应版本表，有效期为公共样本格上的左闭右开区间。"""

    __slots__ = ("_versions",)

    def __init__(self, versions: List[Tuple[int, int, np.ndarray, np.ndarray]]):
        # versions: [(lo, hi, H, ok)]，已排序且互不交叠；ok 标记可校正频点
        self._versions = versions

    @classmethod
    def from_record(cls, entries: Sequence[dict], nfreq: int) -> "ResponseBook":
        versions = []
        for entry in entries:
            lo, hi = (int(v) for v in entry["effective"])
            H = np.zeros(nfreq, dtype=np.complex128)
            ok = np.zeros(nfreq, dtype=bool)
            for f, re_im in enumerate(entry["re_im"][:nfreq]):
                z = complex(re_im[0], re_im[1])
                if z != 0:  # 响应为零或未列出表示该频点不可校正
                    H[f] = z
                    ok[f] = True
            versions.append((lo, hi, H, ok))
        versions.sort(key=lambda v: (v[0], v[1]))
        for prev, nxt in zip(versions, versions[1:]):
            if nxt[0] < prev[1]:
                raise ValueError("响应有效期交叠，无法判定唯一版本")
        return cls(versions)

    def version_for(self, lo: int, hi: int) -> Optional[int]:
        """整个 [lo, hi) 落在唯一版本内才可用；跨两套响应或落空均无效。"""
        for idx, (a, b, _, _) in enumerate(self._versions):
            if a <= lo and hi <= b:
                return idx
        return None

    def coeffs(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._versions[idx][2], self._versions[idx][3]


@dataclass
class _Report:
    rev: int
    start: int  # 公共样本格起点(已加 offset_ticks)
    samples: np.ndarray  # 调用方持有的只读输入，不复制

    @property
    def stop(self) -> int:
        return self.start + self.samples.shape[0]


class _Timeline:
    """同一 (触发, 通道) 下各数据块在公共样本格上的消解结果。"""

    __slots__ = ("segments", "conflicts")

    def __init__(self, segments, conflicts):
        self.segments = segments  # [(lo, hi, samples, offset)]，互不交叠按起点排序
        self.conflicts = conflicts  # frozenset[int]，读数不一致而不可用的时格

    def window_view(self, lo: int, n: int) -> Optional[np.ndarray]:
        """窗口完整且无冲突才返回样本视图，否则 None。"""
        hi = lo + n
        for c in self.conflicts:
            if lo <= c < hi:
                return None
        out = np.empty(n, dtype=np.float64)
        pos = lo
        for seg_lo, seg_hi, samples, offset in self.segments:
            if seg_hi <= pos:
                continue
            if seg_lo > pos:
                return None  # 缺口
            take = min(seg_hi, hi) - pos
            src = offset + (pos - seg_lo)
            out[pos - lo : pos - lo + take] = samples[src : src + take]
            pos += take
            if pos >= hi:
                return out
        return None


def _resolve(reports: List[_Report]) -> _Timeline:
    """把同一 (触发, 通道) 的各块叠到公共时格：一致只占一格，冲突时格不可用。"""
    bounds = sorted({r.start for r in reports} | {r.stop for r in reports})
    segments = []
    conflicts = set()
    for p, q in zip(bounds, bounds[1:]):
        cover = [r for r in reports if r.start <= p and r.stop >= q]
        if not cover:
            continue
        base = cover[0]
        vals = base.samples[p - base.start : q - base.start]
        agree = np.ones(q - p, dtype=bool)
        for other in cover[1:]:
            agree &= other.samples[p - other.start : q - other.start] == vals
        i = 0
        while i < q - p:
            j = i
            while j < q - p and agree[j] == agree[i]:
                j += 1
            if agree[i]:
                segments.append((p + i, p + j, base.samples, p + i - base.start))
            else:
                conflicts.update(range(p + i, p + j))
            i = j
    merged = []
    for lo, hi, samples, offset in segments:
        if merged:
            plo, phi, ps, poff = merged[-1]
            if ps is samples and phi == lo and poff + (phi - plo) == offset:
                merged[-1] = (plo, hi, ps, poff)
                continue
        merged.append((lo, hi, samples, offset))
    return _Timeline(merged, frozenset(conflicts))


class _Acc:
    """某一 (触发, 配对, 频段) 仍开放的累计量；配对按通道名排序定向。"""

    __slots__ = ("count", "denom", "sxx", "syy", "sxy", "support")

    def __init__(self, nwindows: int):
        self.count = 0          # 联合覆盖数：有正权且两路均可校正频点的窗口数
        self.denom = 0.0        # 实际可用窗口频点权重和
        self.sxx = 0.0          # 左通道自功率累计
        self.syy = 0.0          # 右通道自功率累计
        self.sxy = 0j           # 互谱累计：左变换乘右的复共轭
        self.support = _BitSet(nwindows)

    def apply(self, sign: int, widx: int, wsum: float, sxx: float, syy: float, sxy: complex) -> None:
        self.count += sign
        self.denom += sign * wsum
        self.sxx += sign * sxx
        self.syy += sign * syy
        self.sxy += sign * sxy
        (self.support.add if sign > 0 else self.support.discard)(widx)


@dataclass(frozen=True)
class Verdict:
    """某 (触发, 配对, 频段) 的交接判定与谱量。"""

    trigger: str
    pair: Tuple[str, str]
    band: str
    state: str
    coverage: int
    min_windows: int
    support: Tuple[Tuple[int, int], ...]  # 支持窗的窗口号闭区间
    denom: float
    sxx: float
    syy: float
    sxy: complex
    auto_left: Optional[float]
    auto_right: Optional[float]
    cross: Optional[complex]
    coherence: Optional[float]  # 分母为零时 None，表示无定义


@dataclass(frozen=True)
class Snapshot(Verdict):
    """已交给训练的谱快照：凭据不变，谱量与支持窗不变。"""

    credential: str = ""
    seq: int = 0


@dataclass(frozen=True)
class Receipt:
    """上报回执。"""

    status: str  # ingested / duplicate / stale / rejected
    trigger: str
    channel: str
    block_id: str
    rev: int
    affected_windows: Tuple[int, ...] = ()
    note: str = ""


class SpectralOperator:
    """分窗频谱算子：把异步采集块折成联合覆盖的相干谱张量。"""

    def __init__(self, *, nfft, window, starts, rate, offset_ticks, response, bands):
        self.nfft = int(nfft)
        self.win = np.asarray(window, dtype=np.float64)
        if self.win.shape != (self.nfft,):
            raise ValueError("窗系数长度须等于 nfft")
        self.starts = tuple(sorted({int(s) for s in starts}))
        self.rate = float(rate)
        self.offsets = {str(ch): int(v) for ch, v in offset_ticks.items()}
        self.nfreq = self.nfft // 2 + 1
        fold = np.full(self.nfreq, 2.0)  # 单边谱折叠：非直流与非 Nyquist 乘二
        fold[0] = 1.0
        if self.nfft % 2 == 0:
            fold[-1] = 1.0
        # 功率归一：采样率乘窗系数平方和
        self._scale = fold / (self.rate * float(self.win @ self.win))
        self.bands = tuple(BandSpec.from_record(b) for b in bands)
        self._band_by_id = {b.band_id: b for b in self.bands}
        self.responses = {
            str(ch): ResponseBook.from_record(v, self.nfreq) for ch, v in response.items()
        }
        self._reports: Dict[Tuple[str, str, str], _Report] = {}
        self._timelines: Dict[Tuple[str, str], _Timeline] = {}
        self._channels_by_trigger: Dict[str, set] = {}
        self._acc: Dict[Tuple[str, Tuple[str, str], str], _Acc] = {}
        self._snapshots: Dict[str, Snapshot] = {}
        self._seq = 0

    # ---------------------------------------------------------- 上报采入

    def report_block(self, *, trigger, channel, block_id, rev, start, samples) -> Receipt:
        """采入一条块上报；更正只动受影响窗口中含本通道的配对。"""
        trigger, channel, block_id = str(trigger), str(channel), str(block_id)
        rev = int(rev)
        if channel not in self.offsets:
            raise KeyError(f"通道 {channel} 缺少触发时钟 offset_ticks")
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("数据块须为一维样本序列")
        gstart = self.offsets[channel] + int(start)
        key = (trigger, channel, block_id)
        old = self._reports.get(key)
        new_span = (gstart, gstart + data.shape[0])
        if old is not None:
            if rev < old.rev:
                return Receipt("stale", trigger, channel, block_id, rev, note="低版本补报，忽略")
            if rev == old.rev:
                if old.start == gstart and np.array_equal(old.samples, data):
                    return Receipt("duplicate", trigger, channel, block_id, rev, note="同块同版重发，只计一次")
                return Receipt("rejected", trigger, channel, block_id, rev, note="同版内容冲突，保留已采入内容")
            spans = [new_span, (old.start, old.stop)]
        else:
            spans = [new_span]
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        affected = tuple(
            i for i, s in enumerate(self.starts) if s < hi and s + self.nfft > lo
        )
        for w in affected:  # 先按旧时间线撤出本通道相关配对的旧贡献
            self._fold(trigger, w, channel, -1)
        self._reports[key] = _Report(rev, gstart, data)
        self._timelines.pop((trigger, channel), None)
        self._channels_by_trigger.setdefault(trigger, set()).add(channel)
        for w in affected:  # 再按新时间线计入
            self._fold(trigger, w, channel, +1)
        return Receipt("ingested", trigger, channel, block_id, rev, affected)

    # ---------------------------------------------------------- 窗口与折叠

    def _timeline(self, trigger: str, channel: str) -> Optional[_Timeline]:
        key = (trigger, channel)
        if key not in self._timelines:
            reports = [
                r for (t, c, _), r in self._reports.items() if t == trigger and c == channel
            ]
            if not reports:
                return None
            self._timelines[key] = _resolve(reports)
        return self._timelines[key]

    def _window_spectrum(self, trigger: str, channel: str, wstart: int):
        """单窗校正复数谱；数据不完整、跨响应版本或无响应时为 None。"""
        book = self.responses.get(channel)
        if book is None:
            return None
        version = book.version_for(wstart, wstart + self.nfft)
        if version is None:
            return None
        tl = self._timeline(trigger, channel)
        if tl is None:
            return None
        view = tl.window_view(wstart, self.nfft)
        if view is None:
            return None
        H, ok = book.coeffs(version)
        spec = np.fft.rfft(view * self.win)  # 未归一的单边实 DFT
        coeff = np.zeros(self.nfreq, dtype=np.complex128)
        coeff[ok] = spec[ok] / H[ok]  # 各通道 DFT 按响应复数值相除
        return coeff, ok

    def _fold(self, trigger: str, widx: int, channel: str, sign: int) -> None:
        """把本通道在该窗口上涉及各配对的贡献计入(sign=+1)或撤出(-1)。"""
        wstart = self.starts[widx]
        mine = self._window_spectrum(trigger, channel, wstart)
        if mine is None:
            return
        for other in sorted(self._channels_by_trigger.get(trigger, ())):
            if other == channel:
                continue
            res = self._window_spectrum(trigger, other, wstart)
            if res is None:
                continue
            if channel < other:  # 配对按通道名排序，互谱为左乘右的复共轭
                pair, (X, xok), (Y, yok) = (channel, other), mine, res
            else:
                pair, (X, xok), (Y, yok) = (other, channel), res, mine
            for band in self.bands:
                sel = [
                    (f, w)
                    for f, w in band.weights
                    if f < self.nfreq and xok[f] and yok[f]
                ]
                if not sel:
                    continue
                fs = np.fromiter((f for f, _ in sel), dtype=np.intp, count=len(sel))
                ws = np.fromiter((w for _, w in sel), dtype=np.float64, count=len(sel))
                ww = ws * self._scale[fs]
                x = X[fs]
                y = Y[fs]
                sxx = float(ww @ (x.real**2 + x.imag**2))
                syy = float(ww @ (y.real**2 + y.imag**2))
                sxy = complex(ww @ (x * np.conj(y)))
                key = (trigger, pair, band.band_id)
                acc = self._acc.get(key)
                if acc is None:
                    if sign < 0:
                        continue
                    acc = self._acc[key] = _Acc(len(self.starts))
                acc.apply(sign, widx, float(ws.sum()), sxx, syy, sxy)

    # ---------------------------------------------------------- 判定与交付

    def _verdict(self, trigger: str, pair: Tuple[str, str], band_id: str) -> Verdict:
        band = self._band_by_id[band_id]
        canon = tuple(sorted(pair))
        acc = self._acc.get((trigger, canon, band_id))
        count = acc.count if acc else 0
        if count < band.min_windows:
            state = INSUFFICIENT
        elif acc.sxx <= 0.0 or acc.syy <= 0.0:
            state = NO_COMMON_ENERGY
        else:
            state = DELIVERABLE
        if acc is None:
            sxx = syy = denom = 0.0
            sxy = 0j
            support: Tuple[Tuple[int, int], ...] = ()
        else:
            sxx, syy, sxy, denom = acc.sxx, acc.syy, acc.sxy, acc.denom
            support = acc.support.ranges()
        if tuple(pair) != canon:  # 还原请求方向：互谱共轭，自功率换位
            sxx, syy = syy, sxx
            sxy = sxy.conjugate()
        auto_left = sxx / denom if denom > 0 else None
        auto_right = syy / denom if denom > 0 else None
        cross = sxy / denom if denom > 0 else None
        coherence = abs(sxy) ** 2 / (sxx * syy) if sxx > 0 and syy > 0 else None
        return Verdict(
            trigger=trigger,
            pair=tuple(pair),
            band=band_id,
            state=state,
            coverage=count,
            min_windows=band.min_windows,
            support=support,
            denom=denom,
            sxx=sxx,
            syy=syy,
            sxy=sxy,
            auto_left=auto_left,
            auto_right=auto_right,
            cross=cross,
            coherence=coherence,
        )

    def query(self, trigger, pair, band_id) -> Verdict:
        """当前仍开放的结果。"""
        pair = tuple(str(c) for c in pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("配对须为两个不同通道")
        band_id = str(band_id)
        if band_id not in self._band_by_id:
            raise KeyError(f"未知频段 {band_id}")
        return self._verdict(str(trigger), pair, band_id)

    def query_all(self, trigger, pair) -> Dict[str, Verdict]:
        """模型同时读取的若干(可重叠)频段。"""
        return {b.band_id: self.query(trigger, pair, b.band_id) for b in self.bands}

    def claim(self, trigger, pair, band_id) -> Snapshot:
        """训练端领用：冻结当前谱量与支持窗，返回凭据可查的快照。"""
        verdict = self._verdict(str(trigger), tuple(str(c) for c in pair), str(band_id))
        self._seq += 1
        snap = Snapshot(**vars(verdict), credential=f"snap-{self._seq}", seq=self._seq)
        self._snapshots[snap.credential] = snap
        return snap

    def fetch(self, credential: str) -> Snapshot:
        """按原凭据重查旧快照；后续更正不改变其谱量与支持窗。"""
        return self._snapshots[credential]

    # ---------------------------------------------------------- 交接视图

    def status(self) -> dict:
        """交接状态：各触发下的通道、各配对与频带的判定和支持窗身份。"""
        handoffs = []
        for trigger, pair, band_id in sorted(self._acc):
            v = self._verdict(trigger, pair, band_id)
            handoffs.append(
                {
                    "trigger": trigger,
                    "pair": list(v.pair),
                    "band": band_id,
                    "state": v.state,
                    "coverage": v.coverage,
                    "min_windows": v.min_windows,
                    "support": [
                        {"windows": [i, j], "grid": [self.starts[i], self.starts[j] + self.nfft]}
                        for i, j in v.support
                    ],
                    "coherence": v.coherence,
                }
            )
        return {
            "triggers": {t: sorted(cs) for t, cs in self._channels_by_trigger.items()},
            "handoffs": handoffs,
        }

    def workspace_bytes(self) -> dict:
        """工作区估算：只算本算子新分配的留存状态，不含调用方只读输入。"""
        acc = sum(88 + len(a.support._bits) for a in self._acc.values())
        snaps = sum(160 + 16 * len(s.support) for s in self._snapshots.values())
        reports = 96 * len(self._reports)
        timelines = sum(
            64 * len(tl.segments) + 16 * len(tl.conflicts) for tl in self._timelines.values()
        )
        responses = sum(
            16 * self.nfreq * len(book._versions) for book in self.responses.values()
        )
        total = acc + snaps + reports + timelines + responses
        return {
            "accumulators": acc,
            "snapshots": snaps,
            "reports": reports,
            "timelines": timelines,
            "responses": responses,
            "total": total,
        }


def estimate_workspace(*, channels: int, windows_per_trigger: int, bands: int, triggers: int = 1) -> int:
    """按容量口径估算交接工作区字节数(不含调用方持有的只读输入)。"""
    pairs = channels * (channels - 1) // 2
    per_acc = 88 + (windows_per_trigger + 7) // 8
    return pairs * bands * triggers * per_acc
