
# Replace '/path/to/directory' with the directory where you want to download Anaconda
cd /n/data1/hsph/biostat/celehs/lab/va67/for_tcai/downloads

# Download Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# Install Anaconda - follow the on-screen instructions during installation
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# Create a new conda environment named 'inference' with Python 3.10
conda create -n inference python=3.10

# Activate the newly created 'inference' environment
conda activate inference

# Install PyTorch and its dependencies using the specified CUDA version
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install additional Python packages from the requirements.txt file
# Ensure 'requirements.txt' is present in the current directory
pip install -r requirements.txt

# Install fschat with model_worker and webui components
pip install "fschat[model_worker,webui]"

# Update sentence-transformers to the latest version
pip install -U sentence-transformers

# Update FlagEmbedding to the latest version
pip install -U FlagEmbedding
