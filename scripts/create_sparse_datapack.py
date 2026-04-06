#!/usr/bin/env python3
"""Extract cuSPARSE problems from the mathlibs datapack and create a standalone sparse datapack.

Usage:
    uv run python scripts/create_sparse_datapack.py
    uv run python scripts/create_sparse_datapack.py --release=2026-1 --datapack_dir=data/releases/

The output datapack is written to:
    <datapack_dir>/<release>-sparse-problems.tar.gz

After running this script once, use --include=sparse to target only sparse operators:
    uv run compute_eval generate_samples --release=2026-1 --include=sparse --model=<model>
    uv run compute_eval evaluate_functional_correctness --release=2026-1 \\
        --solutions_datapack=<solutions.tar.gz> --mode=docker
"""

import argparse
import copy
from pathlib import Path

from compute_eval.data.data_model import ReleaseVersion
from compute_eval.data.data_pack import ProblemDatapack


def create_sparse_datapack(release: str, datapack_dir: str) -> None:
    datapack_dir_path = Path(datapack_dir)
    release_version = ReleaseVersion(release)

    input_path = datapack_dir_path / f"{release}-problems.tar.gz"
    output_path = datapack_dir_path / f"{release}-sparse-problems.tar.gz"

    if not input_path.exists():
        raise FileNotFoundError(f"Source datapack not found: {input_path}")

    # Read all mathlibs problems, filter to cusparse/* task_ids, re-tag group as "sparse"
    sparse_problems = []
    with ProblemDatapack(input_path, include=["mathlibs"]) as pack:
        for problem in pack.read_items():
            if problem.task_id.startswith("cusparse/"):
                # Copy problem and override group to "sparse"
                data = problem.model_dump()
                data["group"] = "sparse"
                sparse_problems.append(type(problem).model_validate(data))

    if not sparse_problems:
        raise RuntimeError(
            f"No cusparse/* problems found in {input_path}. "
            "Make sure the source datapack contains mathlibs problems."
        )

    ProblemDatapack.create(
        file_path=output_path,
        items=sparse_problems,
        release=release_version,
        description=f"Sparse matrix operator problems (cuSPARSE) extracted from {release} mathlibs group",
    )

    print(f"Created sparse datapack: {output_path}")
    print(f"  Problems extracted : {len(sparse_problems)}")
    print(f"  Source             : {input_path}")
    print()
    print("Next steps:")
    print(f"  # Generate solutions")
    print(f"  uv run compute_eval generate_samples \\")
    print(f"    --release={release} \\")
    print(f"    --include=sparse \\")
    print(f"    --problems_datapack_dir={datapack_dir} \\")
    print(f"    --model=<your-model>")
    print()
    print(f"  # Evaluate correctness")
    print(f"  uv run compute_eval evaluate_functional_correctness \\")
    print(f"    --release={release} \\")
    print(f"    --solutions_datapack=<solutions.tar.gz> \\")
    print(f"    --problems_datapack_dir={datapack_dir} \\")
    print(f"    --mode=docker")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="2026-1", help="Release version (default: 2026-1)")
    parser.add_argument("--datapack_dir", default="data/releases/", help="Directory containing problem datapacks")
    args = parser.parse_args()

    create_sparse_datapack(args.release, args.datapack_dir)


if __name__ == "__main__":
    main()
