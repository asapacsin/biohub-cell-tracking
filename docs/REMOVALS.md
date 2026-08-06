# Cleanup record

After the clean V106 package passed local source/config/CLI/graph/submission/evaluation tests, the
repository was audited for references to the old `biohub_tracker` package. The new package imports
none of it.

Removed tracked groups:

- `src/biohub_tracker/`: classical blob detector, nearest-neighbour-only tracker, custom learned
  architecture, abandoned detector/association training, and old CLI.
- old `configs/*.yaml`: parameters for removed blob, architecture, and training paths.
- old `scripts/` and `scripts/slurm/`: baseline execution, fixture/download helpers, and abandoned
  training orchestration; only the V106 vendoring script remains.
- old tests: coverage for removed implementations; replaced by local clean-pipeline tests.
- old architecture/training notes, experiment logs, and placeholder notebooks.
- obsolete Docker files tied to the removed package.

Raw/sample data, output directories, credentials, weights, submissions, Git history, and uncertain
untracked files were not removed.

