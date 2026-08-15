# Kaggle submission artifacts

Default file for notebook copy to `/kaggle/working/submission.csv`:

- `submission.csv` — classical blob + NN + division baseline (38077 nodes / 32917 edges)
  Currently the strongest local division-calibrated result.

Alternatives produced by the modular architecture (blob detector, no U-Net):

- `submission_architecture_blob_ilp.csv`
- `submission_architecture_blob_learned.csv`

After the Slurm GPU job finishes, also expect:

- `submission_learned.csv` — U-Net + learned association inference (NFS + login sync)

## Notebook usage

Competition is notebook-only; Internet Off for Submit. Prefer regenerating
`submission.csv` inside the notebook from the uploaded code/datasets, or copy
one of these validated CSVs into `/kaggle/working/submission.csv` if the
competition rules allow writing a precomputed file from attached data.
