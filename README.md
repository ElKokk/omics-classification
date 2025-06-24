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

To RUN THE FULL PIPELINE on Windows' Git Bash :


# main pipeline  (here I am benchmarking on 1, 4 and 8 cores for runtime comparison - can be omitted)



------------------------------------------------------------------------------------------------------------------------

#   Benchmark Stage‑1 with several core counts

docker run --rm -v "$(pwd -W)":/omics omics-thesis:dev \
  bash -lc "cd /omics && ./measure.sh 1 4 8"




#   Aggregate wall‑clocks, merge summaries, draw runtime plots

docker run --rm -v "$(pwd -W)":/omics omics-thesis:dev \
  bash -lc "cd /omics && snakemake -s workflow/Snakefile \
           --cores 1 --use-conda host_analysis"



------------------------------------------------------------------------------------------------------------------------


