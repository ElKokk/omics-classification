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


# main pipeline  (here I am benchmarking on 4, 6 and 8 cores for runtime comparison - can be omitted)


for C in 4 6 8
do
  docker run --rm -v "$(pwd -W)":/omics omics-thesis:dev \
    bash -c "cd /omics && ./measure.sh $C"
done



# aggregate & plot the results


docker run --rm -v "$(pwd -W)":/omics omics-thesis:dev \
  bash -c "cd /omics && snakemake -s workflow/Snakefile --cores 1 host_analysis"

