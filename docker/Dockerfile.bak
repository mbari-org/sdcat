FROM python:3.11-slim

LABEL vendor="MBARI"
LABEL maintainer="dcline@mbari.org"
LABEL license="Apache License 2.0"

ARG GIT_VERSION=latest
ARG IMAGE_URI=mbari/sdcat:${GIT_VERSION}
ARG DOCKER_GID=1001
ARG DOCKER_UID=1001

WORKDIR /app

RUN set -ex \
    && addgroup --system --gid ${DOCKER_GID} docker \
    && adduser --system --uid ${DOCKER_UID} --gid ${DOCKER_GID} docker_user \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git gcc

RUN if [ "$GIT_VERSION" != "latest" ]; then \
    git clone -b v${GIT_VERSION} --depth 1 https://github.com/mbari-org/sdcat.git; \
    else \
    git clone --depth 1 https://github.com/mbari-org/sdcat.git;\
    fi

# Install Miniconda since that does a better job of managing torch dependencies than pip
RUN apt-get update && \
    apt-get install -y wget -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /home/docker_user/.conda && \
    bash miniconda.sh -b -p /home/docker_user/miniconda3 && \
    rm -f miniconda.sh

# Install the application dependencies using Miniconda
WORKDIR /app/sdcat
ENV PATH=/home/docker_user/miniconda3/bin:$PATH
ENV HOME=/home/docker_user
RUN conda env create -f environment.yml \
    && apt-get clean \
    && apt-get remove -y gcc git \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R docker_user:docker /app/sdcat && \
    chown -R docker_user:docker /home/docker_user
USER docker_user
WORKDIR /app/sdcat
CMD ["conda", "run", "-n", "sdcat", "python", "sdcat"]