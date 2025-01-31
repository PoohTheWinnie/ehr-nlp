# Define conda environment name
CONDA_ENV = inference

# Define the path to the requirements file
REQUIREMENTS = requirements.txt

# Define the path to the source code directory
SRC_DIR = .

# Default values for GPU parameters
GPU ?= a100
MEM ?= 32
PARTITION ?= gpu # gpu,gpu_quad,gpu_requeue
TIME ?= 1:15:00
JOBID ?= 1092830

# User name
USER ?= wic029

# List all GPU sessions
.PHONY: gpu-info
gpu-info:
	sinfo  --Format=nodehost,available,memory,statelong,gres:40 -p gpu,gpu_quad,gpu_requeue

# List all GPU sessions
.PHONY: gpu-sessions
gpu-sessions:
	squeue -u $(USER)

# GPU session command
.PHONY: start-gpu
start-gpu:
	@echo "Starting GPU session with:"
	@echo "  GPU Type: $(GPU)"
	@echo "  Memory: $(MEM)G"
	@echo "  Time: $(TIME)"
	srun -n 1 --pty -t $(TIME) --mem $(MEM)G -p $(PARTITION) --gres=gpu:1 -w $(GPU) bash

.PHONY: cancel-gpu
cancel-gpu:
	scancel $(JOBID)

# Add this new target
.PHONY: check-nodes
check-nodes:
	@echo "Checking nodes in gpu_quad partition:"
	sinfo -p gpu_quad -o "%n %G %T %C"
	@echo "\nChecking all GPU partitions and their nodes:"
	sinfo -p gpu,gpu_quad,gpu_requeue -o "%P %n %G %T %C"

# Setup and activate conda environment
.PHONY: setup
setup:
	@if ! command -v conda &> /dev/null; then \
		echo "Installing Anaconda..."; \
		cd /home/wic029/ && \
		wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
		bash Anaconda3-2023.09-0-Linux-x86_64.sh -b && \
		eval "$$/home/wic029/anaconda3/bin/conda shell.bash hook"; \
	else \
		echo "Anaconda already installed"; \
	fi
	@if ! conda env list | grep -q "$(CONDA_ENV)"; then \
		echo "Creating new conda environment '$(CONDA_ENV)'..."; \
		conda create -n $(CONDA_ENV) python=3.10 -y && \
		echo ""; \
		echo "To activate the environment, run:"; \
		echo "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)"; \
	else \
		echo "Environment '$(CONDA_ENV)' already exists"; \
		echo ""; \
		echo "To activate the environment, run:"; \
		echo "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)"; \
	fi

# Install packages (run this after activating the environment)
.PHONY: install
install:
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
	pip install -r $(REQUIREMENTS)
	pip install "fschat[model_worker,webui]"
	pip install -U sentence-transformers
	pip install -U FlagEmbedding
	pip install black isort

# Default CUDA version
CUDA_VERSION ?= 124

# Install PyTorch with specific CUDA version
.PHONY: install-torch
install-torch:
	@echo "Installing PyTorch with CUDA $(CUDA_VERSION)"
	pip3 uninstall torch torchvision torchaudio -y
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$(CUDA_VERSION)


# Format code with isort and black
.PHONY: format
format:
	@echo "Sorting imports with isort..."
	isort $(SRC_DIR)
	@echo "Formatting code with black..."
	black $(SRC_DIR)

# Clean up the conda environment
.PHONY: clean
clean:
	@echo "Removing conda environment..."
	@conda env remove --name $(CONDA_ENV) -y
	@echo "Environment cleaned up!"

# Help target to display available commands
help:
	@echo "╔════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                           Makefile Commands                                ║"
	@echo "╚════════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Development Setup:"
	@echo "  make setup         - Create conda environment"
	@echo "  make install-torch - Install PyTorch with CUDA version"
	@echo "  make install       - Install packages (requires activated environment)"
	@echo "  make clean         - Remove the conda environment"
	@echo
	@echo "GPU Management:"
	@echo "  make gpu-info      - List all GPU sessions"
	@echo "  make gpu-sessions  - List all GPU sessions"
	@echo "  make start-gpu     - Start a GPU session"
	@echo "  make cancel-gpu    - Cancel a GPU session"
	@echo
	@echo "Code Quality:"
	@echo "  make format        - Format code with isort and black"
	@echo
	@echo "Other:"
	@echo "  make all          - Same as make setup"