"""Compile small fixed-index gathers matching the hybrid Conv3D extraction failure."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(args):
    from tools.horizon_runtime import ensure_plugin_loader
    ensure_plugin_loader()
    import torch
    from torch import nn
    import hbdk4.compiler as hbc
    from horizon_plugin_pytorch.quantization.hbdk4 import export
    from tools.compile_hybrid_hat import compile_artifact, save

    class Gather(nn.Module):
        def __init__(self, indices, pad_to):
            super().__init__()
            self.count = len(indices)
            if pad_to > self.count:
                indices = torch.cat((indices, torch.zeros(pad_to - self.count, dtype=torch.int32)))
            self.register_buffer("indices", indices)

        def forward(self, rows):
            selected = torch.index_select(rows, 0, self.indices)
            return selected[:self.count]

    class RowSlices(nn.Module):
        def __init__(self, indices):
            super().__init__()
            self.indices = tuple(int(v) for v in indices)

        def forward(self, rows):
            return torch.cat(tuple(rows[index:index + 1] for index in self.indices), dim=0)

    torch.set_num_threads(2)
    torch.manual_seed(831)
    output = (args.output_dir or ROOT / "outputs/hybrid_gather_probe" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = torch.randn(args.rows, args.channels)
    # Include source edges and widely separated rows. Coordinates and values do
    # not enter the graph; only this fixed INT32 lookup and runtime FP32 rows do.
    indices = torch.linspace(0, args.rows - 1, args.active).round().int()
    expected = torch.index_select(rows, 0, indices)
    report = {"source_shape": list(rows.shape), "active_shape": list(expected.shape),
              "indices": indices.tolist(), "march": "nash-p", "cases": {},
              "hbm_execution_verified": False, "board_verified": False}
    summary = output / "report.json"
    save(summary, report)
    for case in args.cases:
        module = (RowSlices(indices) if case == "slices" else
                  Gather(indices.clone(), {"direct": 0, "pad32": 32, "pad64": 64}[case])).eval()
        with torch.no_grad():
            actual = module(rows)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            changed = rows * .7 + .3
            torch.testing.assert_close(module(changed), torch.index_select(changed, 0, indices), atol=0, rtol=0)
            started = time.perf_counter()
            exported = export(module, (rows,), input_names=["dense_rows"], output_names=["active_rows"])
            artifact = output / (case + ".bc")
            hbc.save(exported, str(artifact))
            record = {"pytorch_equivalent": True, "changed_features_equivalent": True,
                      "hbir": str(artifact), "hbir_bytes": artifact.stat().st_size,
                      "export_seconds": time.perf_counter() - started}
            report["cases"][case] = record
            save(summary, report)
            del exported
        # Native compiler crashes stay in a dedicated bounded subprocess. The
        # same monitor is covered by test_compile_hybrid_hat's fake compilers.
        record["compilation"] = compile_artifact(artifact, output / (case + "_hbm"),
                                                  args.timeout, args.rss_limit_mib)
        save(summary, report)
        print("[case]", case, json.dumps(record), flush=True)
    report["any_hbm_compiled"] = any(value["compilation"]["hbm_compiled"] for value in report["cases"].values())
    save(summary, report)
    print("[report]", summary, flush=True)
    return 0 if report["any_hbm_compiled"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rows", type=int, default=45056)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--active", type=int, default=8)
    parser.add_argument("--cases", nargs="+", choices=("direct", "pad32", "pad64", "slices"),
                        default=["direct", "pad32", "pad64", "slices"])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--rss-limit-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if min(args.rows, args.channels, args.active, args.timeout, args.rss_limit_mib) < 1:
        parser.error("Shape dimensions, timeout and memory limit must be positive")
    if args.active > args.rows:
        parser.error("active rows must not exceed source rows")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
