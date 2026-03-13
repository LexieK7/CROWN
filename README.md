# CROWN
A Universal Visual Foundation Model for Computational Cytopathology.


CROWN is a visual foundation model pretrained on over 10 million cytology images.  
It provides transferable and annotation-free feature representations for a wide range of cytological image analysis tasks.

### Highlights
- Pretrained with a DINOv2-style self-supervised framework on large-scale cytology data
- Strong transferability across classification, retrieval, segmentation, detection, and slide-level weakly supervised tasks
- Robust under significant domain shifts, serving as a single scalable backbone for cytology research

The pretrained COIN model weights are available on [Hugging Face](https://huggingface.co/LexieK/Crown).  
You can request access and download the weights directly from the link above.




## Installation Guide

CROWN is built on the DINOv2 architecture and only requires standard PyTorch dependencies to run.
The repository supports Linux, Windows, and macOS (Intel and Apple Silicon).

Some optional libraries (e.g., xformers, mmcv, cuml) rely on CUDA and may not be available on macOS.
These libraries are not required for running the pretrained model.

### 1. Clone the repository

```
https://github.com/LexieK7/CROWN.git
cd CROWN
```

### 2. Linux Installation (CUDA GPU)

Tested on Ubuntu 20.04 + CUDA 12.x

```
conda create -n crown python=3.9
conda activate crown
```

### 3. Install dependencies (cross-platform)

Install the minimal dependencies:
```
pip install -r requirements.txt
```

### 4 Optional CUDA acceleration (Linux GPU only)

For CUDA environments, optional acceleration libraries can be installed.
```
pip install xformers
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11
```

### macOS Installation

macOS does not support CUDA.
Instead, PyTorch uses the MPS backend for GPU acceleration.

Install dependencies directly:

```
pip install \
torch \
torchvision \
numpy \
tqdm \
pillow \
scikit-learn \
einops \
timm \
opencv-python \
matplotlib \
transformers
```
Optional:

```
pip install faiss-cpu
```

Verify MPS support:
```
import torch
print(torch.backends.mps.is_available())
```

If ```True```, the Apple GPU is available.

### Windows Installation

CROWN can also run on Windows with CPU or CUDA GPU.

Create environment：
```
conda create -n crown python=3.9
conda activate crown
```
Install dependencies：
```
pip install -r requirements.txt
```
Optional CUDA acceleration (Windows GPU)
If CUDA is available, you may optionally install acceleration libraries:

```
pip install xformers
```

Note that some RAPIDS libraries (e.g., cuml-cu11) may not be fully supported on Windows.
These libraries are optional and not required for using the pretrained CROWN model.


## Quick Start

After downloading our model weights：

```
from models.vision_transformer import vit_large
import torch

model = vit_large(
    patch_size=16,
    img_size=224,
    init_values=1.0,
    block_chunks=4,
    ffn_layer="swiglufused",
)

state_dict = torch.load("CROWN.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

```

Used for feature extraction:

```
feat = model.forward_features(img)["x_norm_clstoken"]
```


## Image Preprocessing Example

```
from PIL import Image
import torchvision.transforms as T
import torch

img = Image.open("example.png").convert("RGB")

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()
])

img = transform(img).unsqueeze(0)

feat = model.forward_features(img)["x_norm_clstoken"]
print(feat)
```

## Linear Evaluation Example

This repository provides an example script to evaluate the CROWN using linear probing.
The backbone is frozen and a linear classifier is trained on extracted features using 5-fold cross validation.


### Dataset Structure
The evaluation script expects the dataset to follow the PyTorch ImageFolder format.
```
dataset_root/
    task1/
        class1/
            img1.png
            img2.png
        class2/
            img3.png
            img4.png

    task2/
        class1/
        class2/
        class3/
```
Each task folder represents a classification problem, and the script will automatically evaluate all tasks.
Edit the following variables in the script:

```
base_path = "DATASET PATH"
model_path = "MODEL PATH"
save_path = "OUTPUT PATH"
```
Example:
```
base_path = "data/benchmark"
model_path = "weights/CROWN.pth"
save_path = "results/linear_eval.xlsx"
```

Then run:
```
python linear_eval.py
```

This script is provided as a minimal example to demonstrate how to evaluate the CROWN backbone using standard linear probing protocols.


### Running Linear Evaluation



## Tested Environment

The model was trained using:
```
Ubuntu 20.04
Python 3.9
PyTorch 2.0.1
torchvision 0.15
CUDA 12.8
NVIDIA A100 GPU

```

A working environment used for our experiments includes:
```
torch 2.0.1
torchvision 0.15.0
xformers 0.0.21
transformers 4.52.4
timm 0.9.8
scikit-learn 1.6.1
opencv-python 4.12
numpy 1.26

```



## Troubleshooting
```
1 Installation fails on macOS

macOS does not support CUDA libraries such as:

xformers
cuml-cu11

These libraries are optional and should not be installed on macOS.

Please follow the macOS installation instructions above.

2 PyTorch cannot detect GPU

Run:

import torch
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())
```

## Citation
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```

```


If you use the original DIN modOv2el included in this repo, please cite the following papers.

```
@article{Oquab2023DINOv2LR,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Maxime Oquab and Timothee Darcet and Theo Moutakanni and Huy Q. Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russ Howes and Po-Yao (Bernie) Huang and Shang-Wen Li and Ishan Misra and Michael G. Rabbat and Vasu Sharma and Gabriel Synnaeve and Huijiao Xu and Herve Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  journal={ArXiv},
  year={2023},
  volume={abs/2304.07193},
  url={https://api.semanticscholar.org/CorpusID:258170077}
}
```
