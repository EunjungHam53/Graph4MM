# Graph4MM: Weaving Multimodal Learning with Structural Information

This is the official implementation of our paper:

> **Graph4MM: Weaving Multimodal Learning with Structural Information**  
> Xuying Ning\*, Dongqi Fu\*, Tianxin Wei, Wujiang Xu, Jingrui He  
> ICML 2025  
> \* Equal contribution  
> [Paper Link](https://openreview.net/pdf?id=FB2e8PV6qg)

---

## 🧠 Motivation and Method

Real-world multimodal data—such as e-commerce pages or scientific documents—often exhibit rich structural dependencies that go beyond simple image-caption pairs. Entities across modalities interact through contextual dependencies and co-references, forming complex many-to-many relationships. Graphs offer a natural way to model such intra-modal and inter-modal structures.

However, most existing multimodal models either ignore graph structure or treat it as a standalone modality, failing to integrate it effectively with other modalities. This results in fragmented and incomplete understanding. Specifically, two key challenges remain underexplored:

1. How to integrate multi-hop structural information into large foundation models.
2. How to fuse modality-specific signals (text, image, structure) in a principled and unified way.

To address these, we propose **Graph4MM**, a graph-based multimodal learning framework that leverages structural context during both encoding and fusion. Our approach includes:

- **Hop-Diffused Attention**: A structure-aware attention mechanism that injects multi-hop graph information into token-level self-attention via hop-based diffusion and causal masking.
- **MM-QFormer**: A multi-mapping querying transformer that performs flexible and structure-conditioned cross-modal fusion.

Our theoretical and empirical results show that incorporating graph structure improves multimodal understanding and generalization—especially in generative tasks—beyond treating the graph as an isolated modality.

---

## ⚙️ Environment Setup

We recommend using a conda environment with Python 3.8. First, create the environment:

```bash
conda create -n graph4mm python=3.8
conda activate graph4mm
```

Then, install all dependencies:

```bash
pip install -r requirements.txt
```

Make sure to install a version of `torch` compatible with your CUDA version, for example:

```bash
# Example for CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 📦 Data Preparation: WikiWeb2M

We use the [WikiWeb2M dataset](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md) for both generative and discriminative experiments.

1. Create the raw data directory:

```bash
mkdir -p wikiweb2m/raw
```

2. Download the **Train**, **Validation**, and **Test** files from [WikiWeb2M](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md) into `wikiweb2m/raw/`.

3. Create an image folder for storing downloaded image files:

```bash
mkdir -p wikiweb2m/raw/images
```

4. Convert the dataset into PyTorch format by running the preprocessing script:

```bash
cd wikiweb2m
sh process_data.sh
```

This script will download, extract, and preprocess the dataset into usable `.pt` files for training.

---

## 🚀 Run Graph4MM

To train Graph4MM on the generation task (OPT-125M backbone), simply run:

```bash
sh script/train_generation.sh
```

Make sure your data and pretrained checkpoints are correctly set up in the script configuration.

---

We will continue to update this repository with LLaMA backbone, discriminative dataset, and further extensions. Stay tuned!
