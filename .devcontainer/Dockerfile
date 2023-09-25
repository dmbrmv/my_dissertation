FROM pytorch/pytorch
ARG DEBIAN_FRONTEND=noninteractive
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update && apt-get install -y git wget libspatialindex-dev vim git-lfs ttf-mscorefonts-installer
# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.3/zsh-in-docker.sh)" -- \
    -p git \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

RUN pip3 -q install pip --upgrade
RUN conda install -c conda-forge cartopy

RUN mkdir /init
COPY ./requirements.txt /init/requirements.txt
RUN pip install -r /init/requirements.txt

RUN rm ~/.cache/matplotlib -rf

RUN zsh
