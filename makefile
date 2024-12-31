# Define conda environment name
CONDA_ENV = inference

# Define the path to the requirements file
REQUIREMENTS = requirements.txt

# Define the path to the source code directory
SRC_DIR = .

# Default target
.PHONY: setup install format
all: setup install format

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
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make setup   - Create conda environment"
	@echo "  make install - Install packages (run after activating environment)"
	@echo "  make format  - Format code with isort and black"
	@echo "  make clean   - Remove the conda environment"
	@echo "  make all     - Same as make setup"
