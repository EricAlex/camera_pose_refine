FROM hloc:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN pip3 uninstall -y blinker || true  
RUN rm -rf /usr/local/lib/python*/site-packages/blinker*
RUN pip3 install --ignore-installed open3d