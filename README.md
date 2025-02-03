# Building an LLM from Scratch!

A little personal project for me to become familiar with all the underlying mechanisms of LLMs and how to build one from the ground up.

## Content

- Bigram Language Model (bigram.ipynb)

## Initial Setup (for MacOS)

1. Create virtual environment:

```bash
python3.9 -m venv gpu_kernel
source gpu_kernel/bin/activate
pip install matplotlib numpy ipykernel jupyter
pip install torch torchvision torchaudio
```

2. Setup Jupyter Notebook kernel:

```bash
jupyter notebook
```

> In a new terminal window

```bash
python -m ipykernel install --user --name=gpu_kernel --display-name "gpu kernel"
```
