FROM pytorch/pytorch
# avoid question about geographic area
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget libspatialindex-dev \
    vim git-lfs fontconfig fonts-liberation msttcorefonts screen -qq

# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.3/zsh-in-docker.sh)" -- \
    -p git \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

RUN apt-get update -y --fix-missing &&  apt-get upgrade -y

RUN pip3 -q install pip --upgrade
RUN conda install -c conda-forge -y cartopy

WORKDIR /app
COPY  ./ ./

ENV DATA_DIR=/app/data
RUN mkdir "${DATA_DIR}"

RUN mkdir /app/init
COPY ./requirements.txt /app/init/requirements.txt
RUN pip install -r /app/init/requirements.txt
RUN rm ~/.cache/matplotlib -rf

RUN zsh