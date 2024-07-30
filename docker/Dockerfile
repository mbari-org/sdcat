FROM python:3.11-slim

LABEL vendor="MBARI"
LABEL maintainer="dcline@mbari.org"
LABEL license="Apache License 2.0"

RUN apt-get update && apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv

ARG GIT_VERSION=latest
ARG IMAGE_URI=mbari/sdcat:${GIT_VERSION}

ENV PATH="/venv/bin:$PATH"

ENV APP_HOME=/app
ENV HF_HOME=/tmp/transformers_cache

RUN python3 -m pip install --upgrade pip

ADD . .
RUN pip install poetry && poetry build && pip install dist/*.whl && rm -rf dist

RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}
ENV HOME=${APP_HOME}
RUN chmod a+rwx -R ${APP_HOME}
ENTRYPOINT ["sdcat"]