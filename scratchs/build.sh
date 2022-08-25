#!/bin/bash
docker build -t docker-ml-model:v0 -f Dockerfile .
docker run docker-ml-model:v0 python3 runner.py
