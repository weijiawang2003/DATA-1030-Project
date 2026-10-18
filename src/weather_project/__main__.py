"""Command line runner; also see Weather_Project.ipynb."""

import argparse
import json

from .pipeline import STAGES, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Weather project framework; default command only previews the plan")
    parser.add_argument("--stage", choices=STAGES, default="plan", help="Execute only this stage; default plan has no side effects")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--cities", nargs="+", help="City IDs, e.g. boston seattle; default all seven")
    parser.add_argument("--include-svm", action="store_true")
    parser.add_argument("--include-range", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only matching cached API data")
    parser.add_argument("--refresh", action="store_true", help="Replace cached API responses")
    args = parser.parse_args()
    result = run_pipeline(config_path=args.config, mode=args.mode, cities=args.cities,
                 include_svm=args.include_svm, include_range=args.include_range,
                 offline=args.offline, refresh=args.refresh, stage=args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
