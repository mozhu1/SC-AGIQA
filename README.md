# SC-AGIQA

<img width="770" alt="1744720558481" src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" />

# DATASET:
We have provided documents containing descriptive prompts in the folders of AGIQA-1K, AGIQA-3K, and AIGCIQA2023. When using them, you just need to set up the paths according to the figure above and change the dataset paths in the configuration files.

<img width="785" alt="1744721616515" src="https://github.com/user-attachments/assets/77444089-9fc2-49bd-a616-07a686f5baa5" />

# Run the training and validation code.
pip install -r requirements.txt

python main.py

# Generate descriptive prompts
If you want to generate descriptive prompts on your own, we provide a code example named `chat_with_doubao.py` based on the AGIQA-1K dataset. The same principle applies to other datasets. The API can be obtained from this link https://www.volcengine.com/experience/ark
