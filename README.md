This project compares classical, penalised and Super‑Learner classifiers
on multiple high‑dimensional omics datasets.


```bash


CLONE THE REPOSITORY

GITHUB

git clone https://github.com/ElKokk/omics-classification.git
cd omics-classification


GITLAB

git clone https://gitlab.com/Eleftherios_K/omics-classification-thesis.git

cd omics-classification-thesis



CREATE THE ENVIRONMENT AND INSTALL PACKAGES

conda env create -f env/environment.yml
conda activate omics-thesis

RUN THE FULL PIPELINE

snakemake --use-conda -j4


(OR RUN JUST STAGE1)

snakemake --use-conda -j4 mccv_stage1


DOCKERFILE

the prebuilt is on my drive: omics-thesis.tar

use:

docker load -I omics-thesis.tar

you can execute it with:

docker run --rm -v "$PWD":/work -w /work omics-thesis \
           snakemake
