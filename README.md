# Text-Visual Semantic Constrained AI-Generated Image Quality Assessment
[![Platform](https://img.shields.io/badge/Platform-linux-lightgrey?logo=linux)](https://www.linux.org/)

[![Python](https://img.shields.io/badge/Python-3.10.15%2B-orange?logo=python)](https://www.python.org/)

[![Pytorch](https://img.shields.io/badge/PyTorch2.10%2B-brightgree?logo=PyTorch)](https://pytorch.org/)

[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/)   

[ACMMM 2025] This work has been accepted by ACM Multimedia 2025.

>**Abstract** *With the rapid advancements in Artificial Intelligence Generated Image (AGI) technology, the accurate assessment of their quality has become an increasingly vital requirement. Prevailing methods typically rely on cross-modal models like CLIP or BLIP to evaluate text-image alignment and visual quality. However, when applied to AGIs, these methods encounter two primary challenges: semantic misalignment and details perception missing. To address these limitations, we propose Text-Visual Semantic Constrained AI-Generated Image Quality Assessment (SC-AGIQA), a unified framework that leverages text-visual semantic constraints to significantly enhance the comprehensive evaluation of both text-image consistency and perceptual distortion in AI-generated images. Our approach integrates key capabilities from multiple models and tackles the aforementioned challenges by introducing two core modules: the Text-assisted Semantic Alignment Module (TSAM), which leverages Multimodal Large Language Models (MLLMs) to bridge the semantic gap by generating an image description and comparing it against the original prompt for a refined consistency check, and the Frequency-domain Fine-Grained Degradation Perception Module (FFDPM), which draws inspiration from Human Visual System (HVS) properties by employing frequency domain analysis combined with perceptual sensitivity weighting to better quantify subtle visual distortions and enhance the capture of fine-grained visual quality details in images. Extensive experiments conducted on multiple benchmark datasets demonstrate that SC-AGIQA outperforms existing state-of-the-art methods. The code is publicly available at https://github.com/mozhu1/SC-AGIQA.*
## Network Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" alt="Descriptive Alt Text" width="1000">
</p>

## DATASETS

### Download datasets

|Dataset|Link|
| ---------------------------------- | :------------------------------: |
|AGIQA-1K|[download](https://github.com/lcysyzxdxc/AGIQA-1k-Database)|
|AGIQA-3K|[download](https://github.com/lcysyzxdxc/AGIQA-3k-Database.)|
|AIGCIQA2023|[download](https://github.com/wangjiarui153/AIGCIQA2023)|

### Descriptive prompts
We have provided documents containing descriptive prompts in the folders of AGIQA-1K, AGIQA-3K, and AIGCIQA2023. When using them, you just need to set up the paths according to the figure above and change the dataset paths in the configuration files.

```shell 
AGIQ1K/
├── AIGC-1K_answer.csv
└── file/
    ├── image1.png
    ├── image2.png
    └──...
AGIQA3K/
├── data.csv
├── image1.jpg
├── image2.png
└──...
AIGCIQA2023K/
├── merged_output_aigciqa2023.csv
├── DATA/
│   └── MOS/
│       └── mosz1.mat
└── Image/
    ├── subfolder1/
    │   ├── image1.jpg
    │   └──...
    └──...
```

### Generate descriptive prompts(optional)
If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark


## Usage

```shell
Environment: Python 3.10.15 cuda11.8
```

### 1. Code Acquisition
```shell
git clone https://github.com/mozhu1/SC-AGIQA.git
cd ./SC-AGIQA-main
```

### 2. Dependency Installation 
```shell
conda create -n sc_agiqa python=3.10.13 -y
conda activate sc_agiqa
pip install -r requirements.txt
```

### 3. Configuration Setup 
Before running the experiment, modify the `DATA_PATH` in your configuration file (e.g., `/configs/agiqa1k.yaml`) to point to your actual dataset location, then ensure `main.py` is configured to load this specific YAML file for parameter settings.

### 4. Hugging Face Authentication  

```shell
huggingface-cli login --token <your_token>
```

### 5. Train and test

```shell
python main.py
```






