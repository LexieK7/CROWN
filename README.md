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


##  Getting started

Clone this repo:

```
https://github.com/LexieK7/CROWN.git
cd CROWN

conda env create -f conda-extras.yaml

OR

pip install -r requirements.txt -r requirements-extras.txt

```

## Quick usage

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

## Basic Environment
    Linux (Tested on Ubuntu 20.04)
    
    NVIDIA GPU (Tested on A100 ) with CUDA 12.8
    
    Python (3.9)
    
    Torch (2.0.1)
    
    torchvision (0.15.0)

## Citation
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```

```


If you use the original DINOv2 model included in this repo, please cite the following papers.

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
