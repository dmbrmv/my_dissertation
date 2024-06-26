FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
# avoid question about geographic area
ARG ANACONDA_CONTAINER="v23.1.0"
ARG ANACONDA_DIST="Miniconda3"
ARG ANACONDA_PYTHON="py310"
ARG ANACONDA_CONDA="23.1.0"
ARG ANACONDA_OS="Linux"
ARG ANACONDA_ARCH="x86_64"
ARG ANACONDA_FLAVOR="Miniforge3"
ARG ANACONDA_PATCH="4"
ARG ANACONDA_VERSION="${ANACONDA_CONDA}-${ANACONDA_PATCH}"
ARG ANACONDA_INSTALLER="${ANACONDA_FLAVOR}-${ANACONDA_VERSION}-${ANACONDA_OS}-${ANACONDA_ARCH}.sh"
ARG ANACONDA_ENV="base"
ARG ANACONDA_GID="100"
ARG ANACONDA_PATH="/usr/local/anaconda3"
ARG ANACONDA_UID="1000"
ARG ANACONDA_USER="anaconda"
ARG HOST_USER_ID
ARG HOST_USER_NAME
ARG HOST_GROUP_ID
ARG HOST_GROUP_NAME
# Create an arbitrary non-root user; we don't care about its uid
# or other properties

ENV YOUR_ENV=flood \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.3.0 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update -y --fix-missing &&  apt-get upgrade -y

RUN apt-get install software-properties-common -y

# RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN yes 1 | apt-get update && apt-get -y install tzdata \
    bzip2 ca-certificates locales sudo \
    curl wget gcc g++ git screen \
    libpq-dev libgdal-dev gdal-bin python3-gdal

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

# Configure environment
ENV ANACONDA_ENV=${ANACONDA_ENV} \
    ANACONDA_PATH=${ANACONDA_PATH} \
    ANACONDA_GID=${ANACONDA_GID} \
    ANACONDA_UID=${ANACONDA_UID} \
    ANACONDA_USER=${ANACONDA_USER} \
    HOME=/home/${ANACONDA_USER} \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SHELL=/bin/bash

ENV PATH ${ANACONDA_PATH}/bin:${PATH}
# Enable prompt color, generally
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc
# Copy fix-permissions script
COPY scripts/fix-permissions /usr/local/bin/fix-permissions

# Create default user wtih name "anaconda"
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su \
    && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers \
    && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers \
    && useradd -m -s /bin/bash -N -u ${ANACONDA_UID} ${ANACONDA_USER} \
    && mkdir -p ${ANACONDA_PATH} \
    && chown -R ${ANACONDA_USER}:${ANACONDA_GID} ${ANACONDA_PATH} \
    && chmod g+w /etc/passwd \
    && chmod a+rx /usr/local/bin/fix-permissions \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}

# Switch to user "anaconda"
USER ${ANACONDA_UID}
WORKDIR ${HOME}

# Install Anaconda (Miniconda) - https://anaconda.com/
RUN wget --verbose -O ~/${ANACONDA_VERSION}.sh https://github.com/conda-forge/miniforge/releases/download/${ANACONDA_VERSION}/${ANACONDA_INSTALLER} \
    && /bin/bash /home/${ANACONDA_USER}/${ANACONDA_VERSION}.sh -b -u -p ${ANACONDA_PATH} \
    && chown -R ${ANACONDA_USER} ${ANACONDA_PATH} \
    && rm -rvf ~/${ANACONDA_VERSION}.sh \
    && echo ". ${ANACONDA_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate \${ANACONDA_ENV}" >> ~/.bashrc \
    && find ${ANACONDA_PATH} -follow -type f -name '*.a' -delete \
    && find ${ANACONDA_PATH} -follow -type f -name '*.js.map' -delete \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}


ENV PATH ${ANACONDA_PATH}/bin:${PATH}

RUN conda install gdal=3.4.1 cartopy mscorefonts

# Switch back to root
USER root

# Clean Anaconda
RUN conda clean -afy \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}

# Make configuration adjustments in /etc
RUN ln -s ${ANACONDA_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && fix-permissions /etc/profile.d/conda.sh

# Clean packages and caches
RUN apt-get --purge -y remove wget curl \
    && apt-get --purge -y autoremove \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && rm -rvf /home/${ANACONDA_PATH}/.cache/yarn

RUN apt-get update -y --fix-missing && apt-get upgrade -y

WORKDIR /app
COPY  ./ ./

RUN mkdir /app/init
COPY ./requirements.txt /app/init/requirements.txt
RUN pip install -r /app/init/requirements.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN rm ~/.cache/matplotlib -rf

ENV DATA_DIR=/app/geo_data
RUN mkdir "${DATA_DIR}"

RUN bash
# RUN mkdir "$DATA_DIR" && chown user "$DATA_DIR"