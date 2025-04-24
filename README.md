## SC-AGIQA

<img width="770" alt="1744720558481" src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" />

## DATASET:
We have provided documents containing descriptive prompts in the folders of AGIQA-1K, AGIQA-3K, and AIGCIQA2023. When using them, you just need to set up the paths according to the figure above and change the dataset paths in the configuration files.
Environment: Python 3.10
```
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
## Run the code.
*1.Modify the dataset directory in the configuration file and specify the configuration file for the parameters of the main function*
cd ./SC_AGIQA

pip install -r requirements.txt
huggingface-cli login --token <your_token>
python main.py

## Generate descriptive prompts
If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark
