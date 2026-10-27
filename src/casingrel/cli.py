"""Command-line interface.

    casingrel run well.yaml [-o OUT] [--design NAME] [--probabilistic] [--samples N] [--seed S]
        Design check of a reference design: report.md, plots/ and
        design_factors.csv; with --probabilistic, also probability of failure.
    casingrel optimize well.yaml [-o OUT] [--target PF] [--catalog CSV] [--samples N] [--seed S]
                                 [--no-taper] [--no-deterministic-check]
        Cheapest design per string from the pipe catalog, with its report.
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
from .optimize import optimize_well
from .reliability import DEFAULT_MAX_SAMPLES, DEFAULT_SAMPLES, reliability_for_check, shoe_move_probability
from .report import DISCLAIMER, write_report
from .schema import load_well


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="casingrel", description="Probabilistic casing design (student tool).",
                                 epilog=DISCLAIMER)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="design check and report for one input file")
    opt = sub.add_parser("optimize", help="choose the cheapest design from the pipe catalog")
    for p in (run, opt):
        p.add_argument("input", type=Path, help="well input file (.yaml or .json)")
        p.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: reports/<input name>)")
        p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Monte Carlo samples to start with")
        p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                       help="Monte Carlo cap while adding samples to reach a 10%% CoV")
        p.add_argument("--seed", type=int, default=0, help="random seed")
    run.add_argument("--design", default=None, help="reference design to check (default: the first one)")
    run.add_argument("--probabilistic", action="store_true", help="also compute probability of failure")
    opt.add_argument("--target", type=float, default=None, help="Pf target (default: the input's)")
    opt.add_argument("--catalog", type=Path, default=None, help="pipe catalog CSV (default: the input's, "
                                                                "else the starter catalog)")
    opt.add_argument("--no-taper", action="store_true", help="single-pipe strings only")
    opt.add_argument("--no-deterministic-check", action="store_true",
                     help="allow designs that fail the deterministic design factors (spec 6.3)")
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
    out = args.out or Path("reports") / (args.input.stem + ("_optimized" if args.command == "optimize" else ""))
    try:
        if args.command == "run":
            result = check_well(well, design=args.design)
            rel = moves = None
            if args.probabilistic:
                rel = reliability_for_check(result, n=args.samples, seed=args.seed, max_n=args.max_samples)
                moves = shoe_move_probability(well, result.profiles, seed=args.seed)
            path = write_report(result, out, rel, moves)
        else:
            o = optimize_well(well, args.input, args.catalog, args.target, not args.no_deterministic_check,
                              args.samples, args.seed, not args.no_taper, args.max_samples)
            result = check_well(well, profiles=o.profiles, designs=o.designs, design_name="optimized")
            rel = {k: s.chosen.reliability for k, s in o.strings.items() if s.chosen is not None}
            moves = shoe_move_probability(well, o.profiles, seed=args.seed)
            path = write_report(result, out, rel, moves, o)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Cannot run {args.input}: {exc}", file=sys.stderr)
        return 2
    print(_console_summary(result, rel))
    print(f"Report written to {path}")
    return 0


def _console_summary(result, rel=None) -> str:
    lines = [f"{result.well.well.name}: design {result.design_name or '(none, loads only)'}"]
    for g in result.geometries:
        c = result.checks.get(g.name)
        status = "not checked" if c is None else ("PASS" if c.ok else "FAIL")
        pf = ""
        if rel and g.name in rel:
            pf = f"  max Pf {max(r.pf for r in rel[g.name].rows):.1e}"
        label = f"  {c.design.label}" if c is not None else ""
        lines.append(f"  {g.name:<16} shoe {g.shoe_tvd_ft:>8,.0f} ft TVD  {status}{pf}{label}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
