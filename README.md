# Live_ASR_Whisper_Gradio

# Setup

1. Environment

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
```

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
exec "$SHELL"
```

```bash
pyenv install 3.9.17 
pyenv global 3.9.17
```

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Error `Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}`:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

---

Alternative solution:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb 
dpkg -i cuda-keyring_1.0-1_all.deb 
apt update && apt upgrade
apt install libcudnn8 libcudnn8-dev
```