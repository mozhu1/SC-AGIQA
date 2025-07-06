# Text-Visual Semantic Constrained AI-Generated Image Quality Assessment
[ACMMM 2025] This work has been accepted by ACM Multimedia 2025.

## Network Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" alt="Descriptive Alt Text" width="1000">
</p>

## DATASET
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

## Usage

`shell
Environment: Python 3.10.15 cuda11.8
`

### 1.Download and extract the code  
```shell
git clone https://github.com/mozhu1/SC-AGIQA.git
cd ./SC_AGIQA
```

### 2.Install the required packages
```shell
conda create -n sc_agiqa python=3.10.13 -y
conda activate sc_agiqa
pip install -r requirements.txt
```

### 3.Modify the dataset directory in the configuration file and specify the configuration file for the parameters of the main function  

### 4.Log in to Hugging Face in preparation for downloading the ImageReward weights

`shell
huggingface-cli login --token <your_token>
`

### 5.Train and test

`shell
python main.py
`

<h2 align="center">
Generate descriptive prompts
</h2>

If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark

