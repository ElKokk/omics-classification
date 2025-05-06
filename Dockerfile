FROM condaforge/mambaforge:latest


WORKDIR /omics
COPY . .

RUN mamba env create -f env/environment.yml

ENV PATH=/opt/conda/envs/omics-thesis/bin:$PATH

#default command, run the pipeline
ENTRYPOINT ["snakemake", "--cores", "all"]
