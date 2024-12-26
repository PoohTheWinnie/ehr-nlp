# Define the name of the virtual environment directory
VENV_DIR = llama_env

# Define the path to the requirements file
REQUIREMENTS = requirements.txt

# Define the path to the source code directory (for formatting with black)
SRC_DIR = .

# Default target
.PHONY: all
all: setup install

# Create a virtual environment
$(VENV_DIR)/bin/activate:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	python3 -m venv $(VENV_DIR)

# Set up the environment by creating the virtual environment
setup: $(VENV_DIR)/bin/activate

# Install dependencies from requirements.txt
install: $(VENV_DIR)/bin/activate $(REQUIREMENTS)
	@echo "Installing dependencies from $(REQUIREMENTS)..."
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)

# Format code with black
format: $(VENV_DIR)/bin/activate
	@echo "Formatting code with black..."
	$(VENV_DIR)/bin/black $(SRC_DIR)

# Clean up the virtual environment
.PHONY: clean
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Environment cleaned up!"

# Help target to display available commands
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make setup       - Create a virtual environment"
	@echo "  make install     - Install dependencies from requirements.txt"
	@echo "  make format      - Format code with black"
	@echo "  make clean       - Remove the virtual environment"
	@echo "  make all         - Run setup, install, and format commands in sequence"
