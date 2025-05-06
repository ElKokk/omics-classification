This project compares classical, penalised and Super‑Learner classifiers
on multiple high‑dimensional omics datasets.


```bash


CLONE THE REPOSITORY

git clone https://github.com/<you>/omics-classification-thesis.git
cd omics-classification-thesis

CREATE THE ENVIRONMENT AND INSTALL PACKAGES

conda env create -f env/environment.yml
conda activate omics-thesis

RUN THE FULL PIPELINE

snakemake --use-conda -j1


(OR RUN JUST STAGE1)

snakemake --use-conda -j4 mccv_stage1
