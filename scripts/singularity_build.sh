#!/bin/bash

just reinstall
uv export --no-hashes --format requirements-txt >cache/requirements.txt
sudo singularity build cache/container.sif container.def
