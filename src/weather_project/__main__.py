"""Command line preview and explicit data stages."""

import argparse
import json

from .pipeline import STAGES, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Brown weather framework; default is preview only")
    parser.add_argument("--stage", choices=STAGES, default="plan")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
