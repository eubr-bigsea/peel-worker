# Use the official Ubuntu LTS 22.04 as the base image
FROM ubuntu:22.04
LABEL maintainer="Walter Santos <walter@dcc.ufmg.br>, Pedro Henrique <rodrigues.pedro@dcc.ufmg.br>, Lucas Ponce <lucasmsp@dcc.ufmg.br>"

# Set non-interactive mode during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install necessary tools in a single layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        graphviz \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3.10 -m pip install --no-cache-dir --upgrade pip

# Set timezone and working directory
ENV TZ=America/Sao_Paulo
WORKDIR /peel-worker
ENV PYTHONPATH=/peel-worker


# Copy only the requirements file to the working directory
COPY requirements.txt .

# Install Python packages listed in requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .

# Execute workers using ENTRYPOINT
ENTRYPOINT ["celery", "-A", "xai_tasks.app", "worker", "--loglevel=info"]

