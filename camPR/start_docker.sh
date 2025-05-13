#!/bin/bash

xhost local:root && docker run -it --rm -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' --ipc host --privileged -p 8888:8888 --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw  \
-v /home/xin/Downloads:/data/  \
campr:latest \

# docker run -it --rm --gpus all --privileged \
# -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
# -v /home/xin/Downloads:/data/  \
# campr:cuda12.2 \