"""Command-line interface.

    casingrel run well.yaml [-o OUT_DIR] [--design NAME]
        Deterministic run: writes report.md, plots/ and design_factors.csv.
    casingrel synthetic [OUT_DIR]
        Rewrite the bundled synthetic test wells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from . import synthetic
from .design import check_well
from .report import DISCLAIMER, write_report
from .schema import load_well


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="casingrel", description="Probabilistic casing design (student tool).",
                                 epilog=DISCLAIMER)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="deterministic design check and report for one input file")
    run.add_argument("input", type=Path, help="well input file (.yaml or .json)")
    run.add_argument("-o", "--out", type=Path, default=None,
                     help="output folder (default: reports/<input name>)")
    run.add_argument("--design", default=None, help="reference design to check (default: the first one)")
    syn = sub.add_parser("synthetic", help="rewrite the bundled synthetic wells")
    syn.add_argument("out_dir", nargs="?", default=str(synthetic.DEFAULT_EXAMPLES_DIR))
    args = ap.parse_args(argv)

    if args.command == "synthetic":
        for path in synthetic.write_bundled(args.out_dir):
            print(path)
        return 0

    try:
        well = load_well(args.input)
    except ValidationError as exc:
        print(f"Input file {args.input} is not valid:\n{exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Cannot read {args.input}: {exc}", file=sys.stderr)
        return 2
    try:
        result = check_well(well, design=args.design)
    except ValueError as exc:
        print(f"Cannot run {args.input}: {exc}", file=sys.stderr)
        return 2
    out = args.out or Path("reports") / args.input.stem
    path = write_report(result, out)
    print(_console_summary(result))
    print(f"Report written to {path}")
    return 0


def _console_summary(result) -> str:
    lines = [f"{result.well.well.name}: design {result.design_name or '(none, loads only)'}"]
    for g in result.geometries:
        c = result.checks.get(g.name)
        status = "not checked" if c is None else ("PASS" if c.ok else "FAIL")
        lines.append(f"  {g.name:<16} shoe {g.shoe_tvd_ft:>8,.0f} ft TVD  {status}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
