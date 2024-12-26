
# LLaMA 2 Inference with Custom Dataset

This project enables inference using Meta's LLaMA 2 model on custom datasets, such as medical notes or doctor's reports. It utilizes the Hugging Face `transformers` library for model loading and tokenization, as well as PyTorch for computation. The project also includes utilities for extracting embeddings from model outputs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Downloading LLaMA 2 Model Weights](#downloading-llama-2-model-weights)
- [Running the Script](#running-the-script)
- [Arguments](#arguments)
- [Usage](#usage)
- [File Structure](#file-structure)

## Prerequisites

- **Python 3.8+**: Ensure you have Python installed.
- **Hugging Face Access**: You will need access to the LLaMA 2 model on Hugging Face. Visit Meta’s [LLaMA 2 page](https://ai.meta.com/llama/) and request access to download the model weights.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and Activate a Virtual Environment**:
   Create a virtual environment to manage dependencies.
   ```bash
   python3 -m venv llama_env
   source llama_env/bin/activate  # On MacOS/Linux
   ```

3. **Install Dependencies**:
   Install the required Python libraries.
   ```bash
   pip install -r requirements.txt
   ```

## Downloading LLaMA 2 Model Weights

Once you have access, follow these steps to download the LLaMA 2 model:

1. **Request Access**:
   - Visit Meta’s official [LLaMA 2 page](https://ai.meta.com/llama/).
   - Request access and agree to Meta’s terms of use for the model.

2. **Set Up Hugging Face Authentication**:
   - Go to [Hugging Face](https://huggingface.co/) and create an account if you don’t have one.
   - Generate an **access token** by going to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

3. **Download the Model**:
   Use the following code to load the model and tokenizer in your Python script once you have access.
   ```bash
   brew install git-lfs    
   git lfs install 
   cd models/
   git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
   ```

## Running the Script

To execute the main script with LLaMA 2, use the following command:
```bash
python main.py --model-path "models/Llama-2-7b-hf" --model-id "llama2" --data-file data/mimic_smoking.csv --dataset smoking --output-dir ./outputs
```

### Arguments

- `--model-path`: Path or Hugging Face model ID for the LLaMA 2 model (e.g., `"models/llama-2-7b"`).
- `--model-id`: Unique identifier for the model.
- `--data-file`: Path to the CSV file containing the dataset.
- `--dataset`: Dataset type, choose between `smoking` or `cancer`.
- `--output-dir`: Directory where outputs and embeddings will be saved.

### Example

To run the script on a dataset located at `data/mimic_smoking.csv`:
```bash
python main.py --model-path "models/llama-2-7b" --model-id "llama2" --data-file data/mimic_smoking.csv --dataset smoking --output-dir ./outputs
```

## Usage

This script will:
1. Process the input data by generating prompts based on the dataset type.
2. Perform inference with the LLaMA 2 model on each sample in the dataset.
3. Extract embeddings from the model outputs and save them as `.csv` files in the specified output directory.

## File Structure

The project directory is structured as follows:
```
your-repo-name/
├── main.py                # Main script for running inference
├── templates.py           # Prompt engineering templates
├── datasets.py            # Two main datasets
├── requirements.txt       # Dependencies file
├── README.md              # Project documentation
├── data/                  # Directory to store datasets
└── models/                # Directory to store models locally
```
