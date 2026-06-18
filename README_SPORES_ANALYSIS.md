# SPORES Clustering Analysis Workflow

This README documents the analysis for the Calliope-NL SPORES results.

## What Is Included

- `src/calliope_nl_analysis/`: reusable Python helpers for SPORES inventory, feature extraction, clustering, plotting, project-result tables, and distribution packaging.
- `notebooks/01_spores_inventory.ipynb`: compact inventory and manifest checks.
- `notebooks/02_spores_clustering.ipynb`: configurable clustering workbench for selecting a decision variable, clustering method, and parameter preset.
- `notebooks/03_project_results.ipynb`: clean notebook for the selected project-result workflow and small result-table exports.
- `scripts/prepare_spores_distribution.py`: command-line helper for generating the SPORES manifest and optional ZIP archives.
- `metadata/spores_manifest.csv`: generated index of the local SPORES NetCDF files.
- `environment-analysis.yml`: analysis environment with the packages used by the notebooks and helper modules.
- `.gitignore`: ignores local Jupyter noise, generated outputs, ZIP archives, and new `results/spores/*.nc` payloads.

## Structure

```text
src/calliope_nl_analysis/
  spores.py           # filename parsing, manifest creation, inventory checks
  features.py         # capacity and operation-cost feature matrices
  clustering.py       # K-means, DBSCAN, hierarchical clustering helpers
  workbench.py        # configurable clustering presets and diagnostics
  project_results.py  # selected project-result parameters, tables, candidates
  plots.py            # plotting helpers for clear figures
  distribution.py     # manifest and ZIP preparation
notebooks/
  01_spores_inventory.ipynb
  02_spores_clustering.ipynb
  03_project_results.ipynb
metadata/
  spores_manifest.csv
scripts/
  prepare_spores_distribution.py
```

The main workflow is:

1. Build or load a feature matrix.
2. Choose a clustering preset and method.
3. Review tuning diagnostics, validation metrics, centroids, and representative SPORES.
4. Export only small tables or plots needed for the analysis.

## Quick Start

1. Create and activate the analysis environment with `conda env create -f environment-analysis.yml`.
2. Request access to the private SPORES data archives from the project maintainers.
3. Download the required family ZIP archives and `SHA256SUMS.txt` into the same local download folder.
4. (Optional) Verify the downloaded ZIP files with `python scripts/prepare_spores_distribution.py --output-dir <download-folder> --verify-archives`.
5. Extract the required family ZIP archives so the `.nc` files are under `results/spores/`.
6. Run `notebooks/01_spores_inventory.ipynb` to verify the local data inventory.
7. Use `notebooks/02_spores_clustering.ipynb` for configurable analyses, or `notebooks/03_project_results.ipynb` for the selected project-result workflow.

## Clustering Workbench

Use `notebooks/02_spores_clustering.ipynb` for configurable analyses. It currently provides two default presets:

- `flow_cap_default`: capacity workflow with PCA 6, K-means `k=4`, DBSCAN `eps=0.8`/`min_samples=12`, and hierarchical cuts Ward `16`, Complete `5.2`, Single `3.1`.
- `cost_operation_variable_default`: operation-cost time-series workflow with PCA 2, K-means `k=4`, DBSCAN `eps=10`/`min_samples=4`, and hierarchical cuts Ward `100`, Complete `35`, Single `14`.

To tune parameters or add a decision variable, edit `src/calliope_nl_analysis/workbench.py`: update or add an `AnalysisPreset` in `PRESETS`, then update `build_matrix(...)` if the variable needs a new aggregation rule.

Use `notebooks/03_project_results.ipynb` for the selected project-result workflow:

- Capacity clustering on the 9 national `flow_cap` technologies.
- Time-series clustering on national `cost_operation_variable`.
- PCA before clustering: 6 components for capacity and 2 components for time series.
- Ward's method as the interpretation method.
- Outputs: centroids, SPORES family allocation, representative configurations, dispatchable-capacity candidates, PCA loadings, and percentile-based time-series candidates.

## Environment

Create a clean analysis environment:

```bash
conda env create -f environment-analysis.yml
conda activate calliope_NL_analysis
python -m ipykernel install --user --name calliope_NL_analysis --display-name "calliope_NL_analysis"
```

The existing project environment remains untouched. The new environment file is meant for the analysis notebooks and package helpers. The `ipykernel` command registers the environment as a selectable Jupyter kernel.

## Data Availability

The 151 SPORES NetCDF files are not stored in this Git repository because of the high local payload. The archives are kept in a private GitHub release and are available from the project maintainers upon request.

Approved users receive access to the private data repository and can download the requested family ZIP archives from the release assets. Keep `SHA256SUMS.txt` in the same folder as the downloaded ZIP archives if you want to verify them:

```bash
python scripts/prepare_spores_distribution.py --output-dir <download-folder> --verify-archives
```

The same check can also be run from inside the download folder with `shasum -a 256 -c SHA256SUMS.txt`. After verification, extract the archives locally and place only the `.nc` files under `results/spores/`.

The private archive release contains:

- `spores_*.zip`: one ZIP archive per SPORES family.
- `SHA256SUMS.txt`: checksums for verifying downloaded ZIP files.
- `metadata/spores_manifest.csv`

## Related Work

This repository supports the analysis workflow associated with the project work:
[Simplifying MGA-informed energy planning decisions with clustering methods: a case of near-future decisions in the Netherlands](https://www.researchgate.net/publication/403009662_Simplifying_MGA-informed_energy_planning_energy_planning_decisions_with_clustering_methods_-_a_case_of_near-future_decisions_in_Netherlands).
