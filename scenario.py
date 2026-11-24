"""回放采集清单：先采入冻结前的上报，领用冻结请求，再采入后续更正。

用法: python scenario.py data/capture.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from caseio import load_case
from spectral_op import SpectralOperator


def build_operator(records) -> SpectralOperator:
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


def show(tag, v):
    coh = "无定义" if v.coherence is None else f"{v.coherence:.6f}"
    cross = "无" if v.cross is None else f"{v.cross:.6g}"
    print(
        f"{tag} 触发={v.trigger} 配对={v.pair} 频段={v.band} 状态={v.state} "
        f"联合覆盖={v.coverage}/{v.min_windows} 支持窗={v.support}"
    )
    print(
        f"    自功率 左={v.auto_left} 右={v.auto_right} 互谱={cross} 相干={coh} "
        f"(分母权重和={v.denom:.6g})"
    )


def main():
    parser = argparse.ArgumentParser(description="回放合成采集的交接过程")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    records, arrays = load_case(args.path)
    op = build_operator(records)
    blocks = records["blocks"]
    k = records["frozen_after_block"]

    print(f"—— 冻结前上报 {k} 条 ——")
    for b in blocks[:k]:
        r = report(op, arrays, b)
        print(f"{r.status:>9} {r.trigger}/{r.channel}/{r.block_id} rev{r.rev} 影响窗={r.affected_windows} {r.note}")

    fr = records["frozen_request"]
    snap = op.claim(fr["trigger"], fr["pair"], fr["band"])
    print(f"\n—— 训练端领用 {fr}，凭据 {snap.credential} ——")
    show("领用时", snap)

    print(f"\n—— 冻结后上报 {len(blocks) - k} 条 ——")
    for b in blocks[k:]:
        r = report(op, arrays, b)
        print(f"{r.status:>9} {r.trigger}/{r.channel}/{r.block_id} rev{r.rev} 影响窗={r.affected_windows} {r.note}")

    print("\n—— 当前开放结果 ——")
    show("当前  ", op.query(fr["trigger"], fr["pair"], fr["band"]))
    print("\n—— 旧凭据重查(谱量与支持窗不变) ——")
    show("凭据  ", op.fetch(snap.credential))

    print("\n—— 交接状态 ——")
    board = op.status()
    for t, chs in board["triggers"].items():
        print(f"触发 {t}: 通道 {chs}")
    for h in board["handoffs"]:
        coh = "无定义" if h["coherence"] is None else f"{h['coherence']:.6f}"
        print(
            f"  {h['trigger']} {h['pair']} {h['band']:>9} 状态={h['state']:<17} "
            f"联合覆盖={h['coverage']}/{h['min_windows']} 支持窗={h['support']} 相干={coh}"
        )
    print("\n工作区估算(字节):", op.workspace_bytes())


if __name__ == "__main__":
    main()
