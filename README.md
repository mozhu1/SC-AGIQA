# Text-Visual Semantic Constrained AI-Generated Image Quality Assessment  ---


## SC-AGIQA

<center>
  <img width="770" alt="1744720558481" src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" /> 
</center>

## DATASET:
We have provided documents containing descriptive prompts in the folders of AGIQA-1K, AGIQA-3K, and AIGCIQA2023. When using them, you just need to set up the paths according to the figure above and change the dataset paths in the configuration files.
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
'Environment: Python 3.10.15 cuda11.8'

**0.Download and extract the code**  

`cd ./SC_AGIQA`

**1.Install the required packages**  

`pip install -r requirements.txt`

**2.Modify the dataset directory in the configuration file and specify the configuration file for the parameters of the main function**  

**3.Log in to Hugging Face in preparation for downloading the model weights**

`huggingface-cli login --token <your_token>`

**4.Run main.py**

`python main.py`

## Generate descriptive prompts

If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark

