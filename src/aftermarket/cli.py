"""`aftermarket` command line (installed by `pip install -e .`).

    aftermarket audit --model <hf id or path> [--adapter ...] [--quick]   # battery + judges + report card
    aftermarket cards                                                       # leaderboard runs' report cards
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    from aftermarket import audit, card

    ap = argparse.ArgumentParser(prog="aftermarket", description="post-release robustness audits for Tiny Aya fixes")
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="run the attack battery on your model and write a report card",
                       description=audit.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    audit.add_arguments(a)
    c = sub.add_parser("cards", help="report cards for every leaderboard run, from results/tables")
    c.add_argument("--out", default=str(card.CARD_DIR))
    args = ap.parse_args(argv)
    if args.command == "audit":
        if not args.model and not args.adapter:
            a.error("give --model (and/or --adapter)")
        audit.audit(args)
    else:
        card.main(["--out", args.out])


if __name__ == "__main__":
    main()
