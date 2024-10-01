# Live_ASR_Whisper_Gradio

# Setup

1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -f requirements.txt
```

or

```bash
pip install uv
uv python install 3.9
uv venv --python 3.9
source venv/bin/activate
uv pip install -r requirements.txt
```

2. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory

Solution for Ubuntu 22.04

```bash
pip install gdown
gdown 1wj9UU7xjF_1R21ysUxg7o2rULmVVz79-
sudo apt install nvidia-cuda-toolkit
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo find /var/cudnn-local-repo-ubuntu2204-8.9.7.29/ -name '*keyring.gpg' -exec cp {} /usr/share/keyrings/ \;
sudo apt-get update
sudo apt-get install --reinstall libcudnn8 libcudnn8-dev libcudnn8-samples
```

To verify:

```bash
ls /usr/lib/x86_64-linux-gnu/libcudnn* | grep libcudnn_ops_infer.so.8
ls /usr/lib/x86_64-linux-gnu/libcudnn* | grep libcudnn_cnn_infer.so.8
```

---

Alternative solution:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb 
dpkg -i cuda-keyring_1.0-1_all.deb 
apt update && apt upgrade
apt install libcudnn8 libcudnn8-dev
```