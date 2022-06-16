.PHONY: clean env format run_dso_feynman

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = Scientific-Machine-Learning-for-Knowledge-Discovery
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Set up python interpreter environment
env:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
else
		conda env create -f environment.yml
endif
		@echo ">>> New conda env created. Activate with:\nsource activate FYP"

run_dso_feynman:
	python -m src.benchmark -d data/AIFeynman/ai_feynman.csv -m DSO

format:
	python -m black .