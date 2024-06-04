FROM python:3.11-alpine

# Install gcc and other build tools and libraries which are needed for numpy dependencies
RUN apk --no-cache add gcc g++ make blas-dev lapack-dev gfortran musl-dev git libffi-dev cmake

LABEL vendor="MBARI"
LABEL maintainer="dcline@mbari.org"
LABEL license="Apache License 2.0"

ARG GIT_VERSION=latest
ARG IMAGE_URI=mbari/sdcat:${GIT_VERSION}
ARG DOCKER_GID=1001
ARG DOCKER_UID=12065

# Create a user with a specific UID and GID.
RUN addgroup -g ${DOCKER_GID} docker && \
    adduser  -u ${DOCKER_UID} -G docker -D docker_user

# Install in the virtual environment as the docker user.
USER docker_user
WORKDIR /home/docker_user

# Create a virtual environment for the application dependencies.
ENV VIRTUAL_ENV=/home/docker_user/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN if [ "$GIT_VERSION" != "latest" ]; then \
    git clone -b v${GIT_VERSION} --depth 1 https://github.com/mbari-org/sdcat.git; \
    else \
    git clone --depth 1 https://github.com/mbari-org/sdcat.git;\
    fi

WORKDIR sdcat
RUN pip install -r requirements.txt

WORKDIR /tmp
RUN git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git && \
    cd Multicore-TSNE && \
    pip install .  && \
    cd .. && \
    rm -rf Multicore-TSNE \

ENTRYPOINT ["python", "sdcat"]
