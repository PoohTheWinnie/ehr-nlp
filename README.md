

## Table of Contents

- [Setup](#setup)
- [Downloading LLaMA 2 Model Weights](#downloading-llama-2-model-weights)
- [O2 Guide](#o2-guide)
- [Running the Embedding Extraction Script](#running-the-embedding-extraction-script)
- [File Structure](#file-structure)

## Setup

### Environment Setup
```bash
make setup   # Create/activate conda environment and install dependencies
make install # Install all packages (including PyTorch with default CUDA)
```

### PyTorch Installation
```bash
make install-torch                  # Install PyTorch with default CUDA (11.7)
make install-torch CUDA_VERSION=112 # Install PyTorch with CUDA 11.2
make install-torch CUDA_VERSION=116 # Install PyTorch with CUDA 11.6
make install-torch CUDA_VERSION=117 # Install PyTorch with CUDA 11.7
make install-torch CUDA_VERSION=118 # Install PyTorch with CUDA 11.8
```

### GPU Session Management
```bash
make gpu                     # Start GPU session with defaults (A100, 32GB, 1:15:00)
make gpu GPU=v100            # Start with specific GPU type (a100, v100)
make gpu MEM=64              # Start with specific memory in GB
make gpu TIME=2:00:00        # Start with specific time limit
```

### Code Formatting
```bash
make format  # Format code with isort and black
```

### Examples
```bash
# Request A100 GPU with 40GB RAM for 2 hours
make gpu GPU=compute-gc-17-246 MEM=40 TIME=2:00:00

# Install all packages with CUDA 11.2
make install CUDA_VERSION=112
```

### Important Notes
1. After running `make setup`, activate the environment with:
   ```bash
   source $(conda info --base)/etc/profile.d/conda.sh && conda activate inference
   ```
2. Check your CUDA version with `nvidia-smi` before installing PyTorch
3. Use `module load cuda/<version>` to load specific CUDA version


## AG Project Specifications
### Key Concepts for Extraction

#### 1. Driver Mutations
- JAK2
- MPL
- CALR
- Triple negative

#### 2. Essential Thrombocythemia (ET) Symptoms
*Note: List subject to refinement with clinical experts*
- Arthralgia
- Fatigue
- Pruritis
- Satiety
- Night sweats
- Numbness

#### 3. Hydroxyurea-Related Adverse Events
*Note: List subject to refinement with clinical experts*
- Sores/ulcers
- Skin lesions
- Stomach pain
- Constipation
- Diarrhea
- Nausea
- Vomiting

[Full NLP Pipeline Documentation] (https://docs.google.com/presentation/d/1Ze8OEfBJ9iL5HmMw6IbXcW3KJXCY2tu2OcXdY8Ua_Oo/edit?usp=sharing)

## O2 Guide

Model Paths:
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/Llama-2-13b-chat-hf
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/Llama-2-13b-hf
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/Llama-2-7b-chat-hf
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/Llama-2-7b-hf
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/vicuna-13b-v1.5
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/vicuna-7b-v1.5
- /n/data1/hsph/biostat/celehs/LLM_MODELS/Models/WizardLM-13B-V1.2
- /n/data1/hsph/biostat/celehs/lab/hongyi/ehrllm/THUMedInfo/GENIE_en_8b

GPU Instances Info:
- [O2 GPU Resources](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1629290761/Using+O2+GPU+resources)

## Downloading LLaMA 2 Model Weights

Once you have access, follow these steps to download the LLaMA 2 model:

1. **Request Access**:
   - Visit Meta's official [LLaMA 2 page](https://ai.meta.com/llama/).
   - Request access and agree to Meta's terms of use for the model.

2. **Set Up Hugging Face Authentication**:
   - Go to [Hugging Face](https://huggingface.co/) and create an account if you don't have one.
   - Generate an **access token** by going to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

3. **Download the Model**:
   Use the following code to load the model and tokenizer in your Python script once you have access.
   ```bash
   brew install git-lfs    
   git lfs install 
   cd models/
   git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
   ```

## Running the Embedding Extraction Script

Example to execute the main script with LLaMA 2, use the following command:
```bash
python main.py --model-path "models/Llama-2-7b-hf" --model-id "llama2" --data-file data/mimic_smoking.csv --dataset smoking --output-dir ./outputs
```

### Arguments

- `--model-path`: Path or Hugging Face model ID for the LLaMA 2 model (e.g., `"models/llama-2-7b"`).
- `--model-id`: Unique identifier for the model.
- `--data-file`: Path to the CSV file containing the dataset.
- `--dataset`: Dataset type, choose between `smoking` or `cancer`.
- `--output-dir`: Directory where outputs and embeddings will be saved.

### Usage

This script will:
1. Process the input data by generating prompts based on the dataset type.
2. Perform inference with the LLaMA 2 model on each sample in the dataset.
3. Extract embeddings from the model outputs and save them as `.csv` files in the specified output directory.

## File Structure

The project directory is structured as follows:
```
ehr-nlp/
├── data/                                  # Directory to store datasets
├── embedding_extraction.py                # Main script for running inference
├── templates.py                           # Prompt engineering templates
├── genie.py                               # GENIE pipeline code
├── datasets.py                            # Two main datasets
├── requirements.txt                       # Dependencies file
├── makefile                               # Make file utility commands
├── README.md                              # Project documentation
```