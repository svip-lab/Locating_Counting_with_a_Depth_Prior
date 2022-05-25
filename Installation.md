## Installation

###Requirements
- PyTorch >= 0.4.1
- torchvision == 0.2.1
- matplotlib
- opencv-python
- dill
- scipy
- yaml

### Step-by-step installation
```bash
# Firstly, your conda is setup properly with the right environment for that

conda create --n completion python=3.6
conda activate completion


# basic packages
conda install matplotlib dill pyyaml opencv scipy 

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch torchvision=0.2.1 cudatoolkit=9.0

```