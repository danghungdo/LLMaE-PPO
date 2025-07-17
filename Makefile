# Makefile for llmae-ppo

.PHONY: help install check format pre-commit sync

help:
	@echo "Makefile for llmae-ppo"
	@echo "* install          to install all dependencies and install pre-commit"
	@echo "* sync             to sync dependencies from pyproject.toml"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with ruff and isort"
	@echo "* pre-commit       to run the pre-commit check"

PYTHON ?= python
PRECOMMIT ?= uv run pre-commit
RUFF ?= uv run ruff

install:
	uv sync
	uv run pre-commit install

sync:
	uv sync

check: 
	$(RUFF) format --check llmae_ppo
	$(RUFF) check llmae_ppo
pre-commit:
	$(PRECOMMIT) run --all-files

format: 
	uv run isort llmae_ppo
	$(RUFF) format llmae_ppo
	$(RUFF) check --fix llmae_ppo



