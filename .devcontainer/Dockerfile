FROM pytorch/pytorch
RUN apt-get update && apt-get install -y git wget libspatialindex-dev
# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)"
RUN pip3 -q install pip --upgrade
RUN conda install -c conda-forge cartopy
RUN mkdir /init
COPY ./requirements.txt /init/requirements.txt
RUN pip install -r /init/requirements.txt
RUN zsh
