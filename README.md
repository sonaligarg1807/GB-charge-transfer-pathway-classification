# GB Charge-Transfer Pathway Classification

Classification of charge-transfer pathways across a grain boundary (GB) into
primary transport archetypes, based on an *effective link descriptor*
computed from time-resolved electronic coupling data, plus a convergence and
robustness analysis of that classification.

This code was used to classify sampled charge-transfer pathways in a
bicrystal grain-boundary system and to verify that the resulting
classification is stable with respect to the number of sampled pathways and
to the exact thresholds used to define the classes.

## Method summary

For each sampled pathway (a chain of sites/dimers from a source to a target
electrode or region, crossing a grain boundary), the pipeline:

1. Reads a time series of pairwise electronic couplings along the path.
2. Computes, per dimer `i`, the time-averaged absolute coupling `J_i`, a
   coherence measure `C_i = <|T_i|>^2 / <|T_i|^2>`, and the **effective
   link** `E_i = J_i * C_i`.
3. Builds two families of descriptors from the `E_i` (and `J_i`, `C_i`)
   profile along the path:
   - **Symmetric (order-independent)** descriptors — permutation-invariant
     summary statistics of the coupling/coherence distribution.
   - **Non-symmetric (order-aware)** descriptors — bottleneck position,
     bottleneck depth, early/late recovery ratio, mid-path "dip" relative to
     the path ends, and weak-region width, all computed respecting the
     source → target ordering.
4. Uses the order-aware descriptors to assign four secondary label axes
   (`location`, `cause`, `shape`, `recovery`), then merges these into one of
   **four primary labels**:
   - `Source-limited`
   - `Target-limited`
   - `GB-crossing (recovering)`
   - `GB-blocked (non-recovering)`

   (an optional fifth `Diffuse/weakly structured` bucket can be enabled for
   pathways with no clear structure).
5. Selects representative ("medoid") pathways per primary label for
   downstream dynamics/visualization.

The convergence and reliability notebooks then re-run this classification
under repeated random subsampling and threshold perturbation to check that
the primary-label populations are not an artifact of the sample size or the
exact quantile cutoffs chosen.

## Repository layout

```
notebooks/
  01_pathway_classification.ipynb   Main pipeline: descriptors -> primary labels -> representative paths
  02_convergence_and_robustness.ipynb  Subsampling convergence, threshold convergence/sensitivity, bootstrap uncertainty
  03_reliability_heatmaps.ipynb        Label-retention heat maps under bootstrap sampling (+ threshold perturbation)
requirements.txt
```

## Input data

Not included in this repository. Each notebook expects one CSV file per
sampled pathway with the format:

```
sites,1276,1834,...
time,"cpl(1,2)","cpl(2,3)",...
0.0,   0.021, -0.034, ...
0.5,   0.019, -0.031, ...
...
```

- Row 1: `sites` followed by the site IDs along the path (length `n`).
- Row 2: column header, `time` plus one `cpl(i,i+1)` column per dimer
  (`n - 1` columns).
- Remaining rows: the coupling time series.

By default the notebooks look for these files under
`ROOT_DIR.glob("*/TRAJ1/path_couplings_timeseries.csv")` — update `ROOT_DIR`
(and `CSV_PATTERN` if needed) in the configuration cell of each notebook to
point at your own data.

## Running order

1. **`01_pathway_classification.ipynb`** — set `ROOT_DIR` to the directory
   containing your per-pathway coupling CSVs and `OUTPUT_DIR` for the
   results, then run top to bottom. This produces (among other files)
   `paths_order_descriptors_with_archetype_labels.csv` and
   `paths_with_primary_labels.csv`, which are the inputs to the other two
   notebooks.
2. **`02_convergence_and_robustness.ipynb`** — point `ROOT_DIR` /
   `ORDER_DESCRIPTOR_FILE` / `PRIMARY_LABEL_FILE` at the outputs from step 1.
   Produces class-fingerprint plots, subsampling-convergence plots (main
   text) and threshold-convergence / bootstrap-uncertainty / threshold-
   sensitivity plots (SI).
3. **`03_reliability_heatmaps.ipynb`** — point `ORDER_DESCRIPTOR_FILE` at the
   same file. Produces the two label-retention heat maps described below.

All three notebooks have a single "configuration" cell near the top where
paths and analysis parameters (bootstrap counts, quantile thresholds, random
seed, etc.) are set.

## Convergence and robustness checks

`02_convergence_and_robustness.ipynb` and `03_reliability_heatmaps.ipynb`
both test whether the primary classification depends on how many pathways
were sampled, or on the exact quantile thresholds used to define the
classes, by repeated random subsampling and (in notebook 3, part B)
perturbation of the quantile levels themselves.

For a primary class `c`, the reported **label-retention fraction** at
sample size `n` is

$$
S_{n,c} = \frac{1}{BP N_c} \sum_{b=1}^{B} \sum_{p=1}^{P} \sum_{k \in c}
\mathbf{1}\!\left[ L_k^{(b,p,n)} = L_k^{\mathrm{ref}} \right]
$$

where `L_k^ref` is pathway `k`'s primary label from the full-dataset
reference classification, `L_k^(b,p,n)` its label after bootstrap repeat `b`
(and, where applicable, threshold-perturbation repeat `p`) using only `n`
pathways to define the thresholds, `B`/`P` are the number of bootstrap /
perturbation repeats, and `N_c` is the number of pathways in reference class
`c`. A value close to 1 means that class's membership is essentially
unaffected by finite sampling (and, for the combined test, by the exact
threshold definitions).

In the dataset this pipeline was developed against, all four primary
classes reached retention fractions above ~0.87 even when thresholds were
estimated from only 10% of the sampled pathways, and converged toward 1.0
as the sampled fraction increased — i.e. the classification is not an
artifact of sample size or of the specific quantile cutoffs chosen.

## Requirements

```
pip install -r requirements.txt
```

The optional 3D bottleneck-visualization cell at the end of
`01_pathway_classification.ipynb` additionally requires
[MDAnalysis](https://www.mdanalysis.org/) and a `.gro` geometry file; it is
skipped gracefully if either is unavailable.
