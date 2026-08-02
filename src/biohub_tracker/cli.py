from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from biohub_tracker.config import load_config
from biohub_tracker.inspection import (
    format_inspection_report,
    inspect_competition,
    write_inspection_report,
)
from biohub_tracker.pipeline import run_prediction_pipeline
from biohub_tracker.submission import build_submission, validate_submission, write_submission

app = typer.Typer(no_args_is_help=True, help="Biohub cell-tracking data and submission tools.")

CompetitionRoot = Annotated[
    Path,
    typer.Option("--competition-root", exists=True, file_okay=False, resolve_path=True),
]


@app.command("inspect")
def inspect_command(
    competition_root: CompetitionRoot = Path("data/competition"),
    report: Annotated[Path, typer.Option("--report", help="JSON report path.")] = Path(
        "outputs/inspection_report.json"
    ),
) -> None:
    """Discover official files and report metadata without reading image volumes."""
    result = inspect_competition(competition_root)
    write_inspection_report(result, report)
    typer.echo(format_inspection_report(result))
    typer.echo(f"JSON report: {report}")


@app.command("validate-data")
def validate_data_command(competition_root: CompetitionRoot = Path("data/competition")) -> None:
    """Require every authoritative input and readable metadata."""
    result = inspect_competition(competition_root)
    problems = [
        *(f"missing {item}" for item in result["missing_authoritative_inputs"]),
        *result["errors"],
    ]
    if problems:
        typer.echo(format_inspection_report(result))
        raise typer.BadParameter("Data validation failed: " + "; ".join(problems))
    typer.echo(format_inspection_report(result))
    typer.echo("Competition data validation passed.")


@app.command("run")
def run_command(
    competition_root: CompetitionRoot = Path("data/competition"),
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/local_baseline.yaml"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/baseline"),
) -> None:
    """Run prediction when a later milestone supplies a detector and tracker."""
    project_config = load_config(config)
    graphs = run_prediction_pipeline(competition_root, project_config)
    submission = build_submission(graphs, sort_rows=project_config.submission.sort_rows)
    if project_config.submission.strict_validation:
        validate_submission(submission, competition_root)
    destination = write_submission(submission, output / "submission.csv")
    typer.echo(str(destination))


@app.command("validate-submission")
def validate_submission_command(
    submission: Annotated[
        Path, typer.Option("--submission", exists=True, dir_okay=False, resolve_path=True)
    ],
    competition_root: CompetitionRoot = Path("data/competition"),
) -> None:
    """Strictly validate a CSV against official local test data and sample schema."""
    table = pd.read_csv(submission)
    validate_submission(table, competition_root)
    typer.echo("Submission validation passed.")


if __name__ == "__main__":
    app()
