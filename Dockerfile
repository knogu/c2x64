FROM amd64/ubuntu:22.04
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y gcc make git binutils libc6-dev gdb sudo \
    vim build-essential
