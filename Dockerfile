# Use small but Conda‑capable base
FROM condaforge/mambaforge:latest

# 1. copy whole repo
WORKDIR /omics
COPY . .

# 2. build the Conda env *inside* the image
RUN mamba env create -f env/environment.yml

# 3. activate env for all subsequent RUN / CMD
ENV PATH=/opt/conda/envs/omics-thesis/bin:$PATH

# 4. default command = run the pipeline
ENTRYPOINT ["snakemake", "--cores", "all"]
