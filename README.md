# Topological Optimal Transport

This repository contains the code for the paper
[_Beyond Distance: Quantifying Point Cloud Dynamics with Persistent Homology and Dynamic Optimal Transport_](https://arxiv.org/abs/2603.15683).



## Repository Layout

```text
.
|-- src/                    # Local wrappers and shared utilities for this repository
|-- experiments/
|   |-- paper_synthetic/    # Synthetic experiments from the paper
|   |-- dorsogna/           # D'Orsogna dataset experiments
|   `-- stroke/             # Real-data stroke experiments
|-- data/                   # User-supplied external datasets
|-- results/                # Generated outputs
`-- tools/                  # Upstream repositories / external code dependencies
```

## Clone and Setup

Clone the repository and initialize the upstream dependencies:

```bash
git clone https://github.com/YixinWang0126/Point_Cloud_Dynamics.git
cd Point_Cloud_Dynamics
```

Create a Python environment and install the common dependencies used by the
notebooks:

```bash
pip install numpy scipy pandas matplotlib plotly scikit-learn networkx jupyter pot tqdm
```

Depending on which experiment you run, you may also need extra packages such as:

- `gudhi`
- `hypernetx`
- `seaborn`
- `umap-learn`
- `nibabel`
- `nilearn`

Julia is also required for `tools/topf`.

## Data Layout

This repository does not redistribute the original datasets.

- Put the D'Orsogna CSV file at `data/dorsogna/dorsogna.csv` (accessible from the
  [D'Orsogna dataset Dryad page](https://datadryad.org/dataset/doi:10.5061/dryad.91j93#citations))
- Put the stroke dataset under `data/stroke/raw/`

See [data/README.md] for the expected layout.

## Running the Code

Start Jupyter from the repository root:

```bash
jupyter lab
```

Then open the notebooks in:

- `experiments/paper_synthetic/`
- `experiments/dorsogna/`
- `experiments/stroke/`



## Provenance and Attribution

This repository combines local experiment code with upstream reprositories.

The current codebase should be understood as follows:

- [`zsteve/tpot`](https://github.com/zsteve/tpot)
  This repository is the main upstream basis for the experiment structure and
  TPOT-based workflow used here.

- [`zsteve/partitioned_networks`](https://github.com/zsteve/partitioned_networks)
  This dependency is included under `tools/partitioned_networks` as upstream
  third-party code and is used by the TPOT pipeline.

- [`vincent-grande/topf`](https://github.com/vincent-grande/topf)
  This dependency is included under `tools/topf` as upstream third-party code
  and is used for topological feature extraction.

This repository adds its own repository-level organization and local code:

- `experiments/`
  Experiment notebooks for the paper

- `src/`
  Local wrappers and shared helper code used by the notebooks in this repository

- `src/topfmain.py`
  A repository-local compatibility wrapper around the upstream `tools/topf`
  code so the notebooks can run in this layout without editing the upstream
  dependency in place

- `data/` and `results/`
  Repository-local conventions for external datasets and generated outputs

Some files under `src/` are adapted from or closely based on logic used in the
upstream `tpot` repository. In the current repository layout, this mainly
applies to local TPOT-related wrappers and helper code such as:

- `src/tpot.py`
- `src/pd.py`
- `src/topo_util.py`
- `src/topfmain.py`



## Citation

If you use this repository, please cite the associated paper and, where
appropriate, the upstream projects it depends on.
