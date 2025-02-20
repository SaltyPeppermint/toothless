#!/bin/bash

just reinstall
uv export --no-hashes --format requirements-txt >cache/requirements.txt
apptainer build --force python_container.sif python_container.def
