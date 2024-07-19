FROM ubuntu:22.04

LABEL vendor="MBARI"
LABEL maintainer="dcline@mbari.org"
LABEL license="Apache License 2.0"

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_VERSION=latest
ARG IMAGE_URI=mbari/sdcat:${GIT_VERSION}

ENV APP_HOME=/app
ENV PYTHONPATH=${APP_HOME}/sdcat
ENV HF_HOME=/tmp/transformers_cache
WORKDIR ${APP_HOME}
RUN if [ "$GIT_VERSION" != "latest" ]; then \
    python3.11 -m pip install sdcat:${GIT_VERSION}; \
    else \
    python3.11 -m pip install sdcat; \
    fi

ENTRYPOINT ["sdcat"]
