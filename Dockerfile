FROM condaforge/mambaforge:latest
WORKDIR /omics

# --------------------------------------------------------------------- system
RUN apt-get update && apt-get install -y --no-install-recommends \
      time tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------- conda
COPY env/environment.yml env/environment.yml
RUN mamba env create -f env/environment.yml -n omics-thesis
RUN conda config --system --set channel_priority strict
ENV PATH=/opt/conda/envs/omics-thesis/bin:$PATH

# --------------------------------------------------------------------- helper
COPY measure.sh /usr/local/bin/measure.sh
RUN chmod +x /usr/local/bin/measure.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
