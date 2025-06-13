FROM condaforge/mambaforge:latest

WORKDIR /omics

RUN apt-get update && apt-get install -y time

COPY env/environment.yml env/environment.yml

RUN mamba env create -f env/environment.yml
RUN conda config --system --set channel_priority strict

ENV PATH=/opt/conda/envs/omics-thesis/bin:$PATH

ENTRYPOINT ["snakemake", "-s", "workflow/Snakefile", "--cores", "all"]
