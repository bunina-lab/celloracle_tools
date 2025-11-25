# CellOracle Pipeline Runner

Lightweight wrappers for installing CellOracle reference genomes and running the GRN inference workflow used in the Bunina lab. The repository provides two entrypoints:

- `run_install_genome.py`: download and install one of the supported genomes into your local genome_dir.
- `run_celloracle_inference.py`: launch the inference pipeline end-to-end from a YAML configuration file via `CellOraclePipeline`.

## Installation

```bash
git clone git@github.com:bunina-lab/celloracle_tools.git
```
or if you haven't got SSH key in your github account:

```bash
git clone https://github.com/bunina-lab/celloracle_tools.git
```

## Environment Setup

CellOracle pulls in heavy scientific dependencies (Scanpy, PyBedtools, PySam, etc.), so using Conda/Mamba is strongly recommended.

1. **Create a clean environment**
   ```bash
   mamba / conda create -n celloracle python=3.10 pip -c conda-forge -c bioconda
   mamba / conda activate celloracle
   ```
2. **Install Python dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
   The `conda_env_requirements.txt` file lists the full package set currently used on Bunina lab machines if you need to audit exact versions.

3. **System utilities**  
   Some Python wheels expect external tools to be present. Ensure `bedtools`, `samtools`, `gzip`, and a C/C++ toolchain (`gcc`, `g++`, `make`) are available on PATH before installing.


## Configuring Inputs

Use `config.yaml` as a template and adjust:

- `rna_h5ad`: AnnData file with RNA counts (raw counts recommended).
- `peak_names_file`: text file with one genomic peak per line (`chr_start_end`).
- `peak_coaccess_path`: TSV with three columns (`peak1`, `peak2`, `weight`).  (You can calculate this by pipe: [circe](https://github.com/bunina-lab/peak_coaccessibility))
- `TG2TF_json_path`: JSON mapping target genes to transcription factors.
- `genome_dir` / `reference_dir`: location and name of the genome installed via `run_install_genome.py`.
- `cluster_column`, `embedding`, `raw_counts`: AnnData metadata settings.
- `tf_binding_frp`, `motif_filtering_method`, etc.: motif/GRN hyper-parameters.

Resolved paths can be absolute or relative; absolute paths are recommended for reproducibility. The script copies the effective configuration to `used_config.yaml` inside each run directory.

## Installing Reference Genomes

The installer only downloads genomes that CellOracle supports (list defined near the top of `run_install_genome.py`). Example:

```bash
python run_install_genome.py \
  --genome_dir /path/to/genomes/celloracle_refs \
  --genome_name hg38
```

- `--genome_dir`: directory where genome FASTA/motif resources will be cached.
- `--genome_name`: one of the choices encoded in the script (human, mouse, zebrafish, fly, worm, etc.).

The script checks whether the requested genome already exists and exits early if so. The installation may take tens of minutes depending on download speed.

## Running CellOracle Inference

Once genomes and inputs are ready, launch the inference workflow:

```bash
python run_celloracle_inference.py \
  --config /path/to/experiment_config.yaml \
  --n_cpu 16
```

Key behaviors:

- The script loads your YAML config, resolves paths, and creates an output run folder at `output_dir/run_name` (defaults to ISO date if `run_name` is omitted).
- RNA `AnnData` is loaded from `rna_h5ad`. If `raw_counts` is `true` and the AnnData object contains a `raw_count` layer, the script replaces `adata.X` with that layer for CellOracle training.
- Peak co-accessibility is read from `peak_coaccess_path` and validated to have exactly three columns.
- `CellOraclePipeline` steps are executed in order: base GRN construction from motifs, Oracle initialization, and GRN inference.
- The `--n_cpu` CLI argument overrides the value in the config file.

Outputs include GRN tables, diagnostics, plots, and the `used_config.yaml` snapshot. See `sarah_test_output/` for an example run layout.

## Tips & Troubleshooting

- **AnnData column availability**: ensure `cluster_column` exists in `adata.obs`; the script aborts early if it cannot find this column.
- **Peak metadata**: `peak_names_file` must align with the peaks used in your co-accessibility table; mismatches lead to empty GRNs.
- **Disk usage**: genome installations can exceed multiple GB; keep `genome_dir` on high-capacity storage.
- **Reproducibility**: track both the exact config file and the `requirements.txt` commit hash when sharing results.

## Getting Help

- Check `lib/process_celloracle.py` for more details on each pipeline stage.

