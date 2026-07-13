import argparse
import json
import sys

from scientific_readiness_runtime import build_suite_readiness_report
from scientific_readiness_runtime import format_suite_readiness_lines
from scientific_readiness_runtime import list_scientific_workflow_ids


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Report scientific workflow readiness for ProtLigInteract.")
    parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="Limit the report to one or more workflow ids.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    parser.add_argument("--list", action="store_true", help="List known workflow ids and exit.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any selected workflow is missing required dependencies.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if bool(args.list):
        for item in list_scientific_workflow_ids():
            print(item)
        return 0
    try:
        report = build_suite_readiness_report(workflow_ids=list(args.workflow or []))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if bool(args.json):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for line in format_suite_readiness_lines(report):
            print(line)
    if bool(args.strict) and (not bool(report.get("ok", False))):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
