FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git init -q && printf 'before\n' > answer.txt && git add answer.txt && git -c user.name=fixture -c user.email=fixture@localhost commit -qm initial
