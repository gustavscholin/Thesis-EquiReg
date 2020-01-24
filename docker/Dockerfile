FROM tensorflow/tensorflow:2.0.0-gpu-py3
ARG USER_ID=1000
ARG USER_NAME=admin
COPY docker/requirements.txt /tmp/
RUN pip install pip -U && \
    pip install --requirement /tmp/requirements.txt && \
    apt-get update && \
    apt-get install -y sudo tmux less vim
RUN useradd ${USER_NAME} -u ${USER_ID} --create-home --shell /bin/bash && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER_NAME}
WORKDIR /home/${USER_NAME}/src
USER ${USER_NAME}
