#!/usr/bin/env python3

"""Plot one or more CSV files produced by benchmark_gemm."""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DERIVED_X_COLUMNS = {"shape", "flops", "output_elements"}

AXIS_LABELS = {
    "shape": "Matrix shape (M x N x K)",
    "m": "M",
    "n": "N",
    "k": "K",
    "flops": "Floating-point operations (2MNK)",
    "output_elements": "Output elements (MN)",
    "kernel_min_ms": "Kernel time [ms]",
    "kernel_median_ms": "Kernel time [ms]",
    "kernel_p90_ms": "Kernel time [ms]",
    "kernel_gflops": "Kernel throughput [GFLOP/s]",
    "pipeline_median_ms": "Pipeline time [ms]",
    "pipeline_p90_ms": "Pipeline time [ms]",
    "pipeline_effective_gflops": "Pipeline throughput [GFLOP/s]",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CSV output from the GEMM benchmark."
    )
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument(
        "--x",
        default="shape",
        help="CSV column or derived value: shape, flops, output_elements",
    )
    parser.add_argument(
        "--y",
        nargs="+",
        default=["kernel_gflops", "pipeline_effective_gflops"],
        help="one or more numeric CSV columns",
    )
    parser.add_argument(
        "--series",
        default="implementation",
        help="column used to split data into separate series; use 'source' "
             "to distinguish input files",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="keep rows with an exact column value; may be repeated",
    )
    parser.add_argument(
        "--square-only",
        action="store_true",
        help="keep only rows satisfying M=N=K",
    )
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="print available columns and exit",
    )
    return parser.parse_args()


def read_benchmark(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    metadata: dict[str, str] = {}
    csv_lines: list[str] = []

    with path.open("r", encoding="utf-8") as benchmark_file:
        for line in benchmark_file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                entry = stripped[1:].strip()
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            csv_lines.append(line)

    if not csv_lines:
        raise ValueError(f"{path}: no CSV data found")

    rows = list(csv.DictReader(io.StringIO("".join(csv_lines))))
    if not rows:
        raise ValueError(f"{path}: CSV contains no benchmark rows")

    for row in rows:
        row["source"] = path.stem

    return rows, metadata


def parse_filters(expressions: Iterable[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(
                f"invalid filter {expression!r}; expected COLUMN=VALUE"
            )
        column, value = expression.split("=", 1)
        if not column:
            raise ValueError(f"invalid filter {expression!r}")
        filters.append((column, value))
    return filters


def filtered_rows(
    rows: Iterable[dict[str, str]],
    filters: list[tuple[str, str]],
    square_only: bool,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []

    for row in rows:
        if square_only and not (row.get("m") == row.get("n") == row.get("k")):
            continue
        if any(row.get(column) != value for column, value in filters):
            continue
        selected.append(row)

    return selected


def numeric(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except KeyError as error:
        raise ValueError(f"unknown column {column!r}") from error
    except ValueError as error:
        raise ValueError(
            f"column {column!r} contains non-numeric value {row[column]!r}"
        ) from error

    if not math.isfinite(value):
        raise ValueError(f"column {column!r} contains non-finite value")
    return value


def x_value(row: dict[str, str], column: str) -> str | float:
    if column == "shape":
        return f"{row['m']}x{row['n']}x{row['k']}"
    if column == "flops":
        return 2.0 * numeric(row, "m") * numeric(row, "n") * numeric(row, "k")
    if column == "output_elements":
        return numeric(row, "m") * numeric(row, "n")
    return numeric(row, column)


def human_label(column: str) -> str:
    return AXIS_LABELS.get(column, column.replace("_", " ").title())


def default_title(metadata: list[dict[str, str]]) -> str:
    gpus = {entry.get("gpu") for entry in metadata if entry.get("gpu")}
    if len(gpus) == 1:
        return f"CUDA GEMM benchmark — {next(iter(gpus))}"
    return "CUDA GEMM benchmark"


def main() -> int:
    arguments = parse_arguments()

    all_rows: list[dict[str, str]] = []
    all_metadata: list[dict[str, str]] = []
    for path in arguments.csv_files:
        rows, metadata = read_benchmark(path)
        all_rows.extend(rows)
        all_metadata.append(metadata)

    columns = set(all_rows[0]) | DERIVED_X_COLUMNS
    if arguments.list_columns:
        for column in sorted(columns):
            print(column)
        return 0

    filters = parse_filters(arguments.where)
    rows = filtered_rows(all_rows, filters, arguments.square_only)
    if not rows:
        raise ValueError("no benchmark rows remain after filtering")

    required_columns = {arguments.series, *arguments.y}
    unknown = required_columns - set(all_rows[0])
    if unknown:
        raise ValueError(f"unknown CSV columns: {', '.join(sorted(unknown))}")
    if arguments.x not in columns:
        raise ValueError(f"unknown x-axis column {arguments.x!r}")

    if arguments.output is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[tuple[str | float, float]]] = defaultdict(list)
    for row in rows:
        series = row[arguments.series]
        for y_column in arguments.y:
            grouped[(series, y_column)].append(
                (x_value(row, arguments.x), numeric(row, y_column))
            )

    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    multiple_series = len({key[0] for key in grouped}) > 1

    for (series, y_column), points in grouped.items():
        if arguments.x != "shape":
            points.sort(key=lambda point: float(point[0]))

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        metric_label = human_label(y_column)
        label = f"{series}: {metric_label}" if multiple_series else metric_label
        axis.plot(x_values, y_values, marker="o", linewidth=1.8, label=label)

    if arguments.log_x:
        if arguments.x == "shape":
            raise ValueError("--log-x cannot be used with categorical shape labels")
        axis.set_xscale("log")
    if arguments.log_y:
        axis.set_yscale("log")

    axis.set_xlabel(human_label(arguments.x))
    if len(arguments.y) == 1:
        axis.set_ylabel(human_label(arguments.y[0]))
    else:
        axis.set_ylabel("Value")
    axis.set_title(arguments.title or default_title(all_metadata))
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    if arguments.x == "shape":
        axis.tick_params(axis="x", rotation=30)

    if arguments.output is None:
        plt.show()
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.output, dpi=180)
        print(arguments.output)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

