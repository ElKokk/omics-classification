This project compares classical, penalised and the Super‑Learner classifier on multiple high‑dimensional omics datasets.


```bash


CLONE THE REPOSITORY

GITHUB

git clone https://github.com/ElKokk/omics-classification.git
cd omics-classification


GITLAB

git clone https://gitlab.com/Eleftherios_K/omics-classification-thesis.git

cd omics-classification-thesis


CONTAINIRIZED WORKFLOW via DOCKER
CREATE THE ENVIRONMENT AND INSTALL PACKAGES



- The workflow is driven by `workflow/Snakefile` and `workflow/config.yaml`.

 - A Conda/Mamba environment (`omics-thesis`) is used to run Snakemake and tools.


 - Runtime benchmarking is handled by `measure.sh` which runs the pipeline with different core counts and saves figures and tables under `Figures/` and `results/`.

---




1) Install Docker

- **Windows:** Install Docker Desktop and ensure “Use the WSL 2 based engine” is enabled.

- **Linux:** Install Docker via distribution and add your user to the `docker` group.


2) Build the image

From the repository root:

```bash
docker build -t omics-thesis:latest .
```


Run Snakemake in the container



**Linux/macOS:**


```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  omics-thesis:dev \
  snakemake --cores all \
    --snakefile workflow/Snakefile \
    --configfile workflow/config.yaml \
    --config n_splits=5
```


**Windows Git Bash:**


```bash
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" \
docker run --rm \
  -v "$(pwd -W):/workspace" \
  -w /workspace \
  omics-thesis:dev \
  snakemake --cores all \
    --snakefile workflow/Snakefile \
    --configfile workflow/config.yaml \
    --config n_splits=5

```

Replace `--cores all` with a number 







Outputs appear in  `results/` and `Figures/` folders.

---

## Native Linux  with Micromamba/Conda

1) System packages

```bash
sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends \
  libcurl4-openssl-dev libssl-dev libxml2-dev libzstd1 libharfbuzz-dev libfribidi-dev
```



2) Create the environment

From the repo root:
```bash
micromamba create -y -n omics-thesis -f env/environment.yml
```



3) Run the pipeline

```bash
micromamba run -n omics-thesis \
  snakemake --cores "$(nproc)" \
    --snakefile workflow/Snakefile \
    --configfile workflow/config.yaml \
    --config n_splits=5
```

Use `--cores N` with a specific number

4) Benchmarking on Linux

```bash
bash measure.sh CRC_microbiome 1 4 16 32
```

---

## Configuration

The pipeline uses `workflow/config.yaml` for its parameters and inputs. Open it and adjust any dataset paths, filenames, and options to match your environment. You can override parameters at the command line via `--config`


```bash
--config n_splits=10
```

