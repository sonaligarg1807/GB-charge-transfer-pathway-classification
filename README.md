# gbctp: GB Charge-Transfer Pathway Classification

Classifies charge-transfer pathways across a grain boundary (GB) into
primary transport archetypes, from time-resolved electronic coupling data,
and includes convergence/reliability checks for that classification.

For each sampled pathway (a chain of sites/dimers from a source to a target
region, crossing a grain boundary), the pipeline computes a
per-dimer *effective link* `E_i = J_i * C_i` (time-averaged coupling `J_i`
weighted by a coherence measure `C_i`), builds symmetric and order-aware
descriptors from the resulting profile, and classifies each pathway into one
of four primary labels: `Source-limited`, `Target-limited`,
`GB-crossing (recovering)`, `GB-blocked (non-recovering)` (with an optional
fifth `Diffuse/weakly structured` bucket).

## Installation

```
pip install -e .            # core package (numpy, pandas, scikit-learn)
pip install -e ".[notebooks]"  # + matplotlib, seaborn, jupyter, for the notebooks
pip install -e ".[geometry]"   # + MDAnalysis, for the optional 3D bottleneck plot
```

## Repository layout

```
gbctp/                       Installable package -- all computation lives here
  io.py                      Reading per-pathway coupling CSVs
  descriptors.py             Per-dimer J/C/E metrics + symmetric/order-aware descriptors
  labeling.py                Threshold computation, archetype labels, primary-label merging
  representatives.py         Representative ("medoid") pathway selection
  pipeline.py                End-to-end classification pipeline (PipelineConfig, run_pipeline)
  reliability.py             Subsampling convergence, threshold sensitivity, bootstrap heat maps
  geometry.py                Optional .gro geometry loading (needs MDAnalysis)

scripts/                     CLI wrappers around gbctp, for non-interactive/batch use
  run_classification.py      Runs gbctp.pipeline end to end
  run_convergence.py         Runs the convergence/sensitivity checks in gbctp.reliability
  run_reliability_heatmaps.py  Runs the bootstrap/threshold-perturbation heat-map checks

notebooks/                   Plotting only -- import gbctp, configure inputs, visualize
  01_pathway_classification.ipynb
  02_convergence_and_robustness.ipynb
  03_reliability_heatmaps.ipynb
```

Use the package/scripts directly for batch runs or your own analysis code;
use the notebooks when you want the figures. Every function has a docstring
describing its inputs/outputs -- start from `gbctp/pipeline.py` and
`gbctp/reliability.py` for the two entry points.

## Input data

Not included in this repository. Each pathway is one CSV with the format
(see `gbctp/io.py:read_path_csv`):

```
sites,1276,1834,...
time,"cpl(1,2)","cpl(2,3)",...
0.0,   0.021, -0.034, ...
0.5,   0.019, -0.031, ...
...
```

- Row 1: `sites` followed by the site IDs along the path, source -> target
  (length `n`).
- Row 2: column header, `time` plus one `cpl(i,i+1)` column per dimer
  (`n - 1` columns).
- Remaining rows: the coupling time series (signed, arbitrary units).

A `root_dir` / `csv_pattern` pair (e.g.
`"*/TRAJ1/path_couplings_timeseries.csv"`) locates one such file per
pathway; see the `TODO` placeholders in `PipelineConfig` / the notebooks'
configuration cells.

## Usage

Python:

```python
from pathlib import Path
from gbctp.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    root_dir=Path("/path/to/pathway_csvs"),
    output_dir=Path("/path/to/outputs"),
    csv_pattern="*/TRAJ1/path_couplings_timeseries.csv",
)
result = run_pipeline(config)
print(result.summary_primary)
```

CLI:

```
python scripts/run_classification.py --root-dir /path/to/pathway_csvs --output-dir /path/to/outputs
python scripts/run_convergence.py --order-descriptor-file /path/to/outputs/paths_order_descriptors_with_archetype_labels.csv --output-dir /path/to/outputs/figures
python scripts/run_reliability_heatmaps.py --order-descriptor-file /path/to/outputs/paths_order_descriptors_with_archetype_labels.csv --output-dir /path/to/outputs/results
```

Then open the notebooks, set the input paths in their configuration cell,
and run top to bottom for the figures.

## Running order

1. `01_pathway_classification.ipynb` (or `run_classification.py`) reads the
   pathway CSVs and produces `paths_order_descriptors_with_archetype_labels.csv`
   and `paths_with_primary_labels.csv`, which are the inputs to the other two.
2. `02_convergence_and_robustness.ipynb` (or `run_convergence.py`) checks
   convergence of class fractions and thresholds with sample size, and
   sensitivity to the exact quantile cutoffs.
3. `03_reliability_heatmaps.ipynb` (or `run_reliability_heatmaps.py`)
   produces label-retention heat maps under bootstrap sampling (+ optional
   threshold perturbation).

## License

MIT, see [LICENSE](LICENSE).
