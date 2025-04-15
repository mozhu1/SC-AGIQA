# SC-AGIQA
SC-AGIQA
<img width="770" alt="1744720558481" src="https://github.com/user-attachments/assets/b5bfa381-4c95-4e88-8fa6-0d8a59cb2100" />


AGIQA1K/
├── AIGC-1K_answer.csv      
└── file/                   
    ├── image1.png         
    └── image2.png
    └── ...  
AGIQA3K/
├── data.csv                # 必须：包含 name, mos_quality, prompt, answer 列
├── image1.jpg              # 图片文件（名称与 data.csv 中的 name 列严格对应）
├── image2.png
└── ...

AIGCIQA2023K/
├── merged_output_aigciqa2023.csv    # 必须：包含 name, prompt, answer 列
├── DATA/
│   └── MOS/
│       └── mosz1.mat                # 必须：包含 MOSz 数组的MATLAB文件
└── Image/
    ├── subfolder1/                  # 允许子目录（CSV中name需包含相对路径）
    │   ├── image1.jpg
    │   └── ...
    └── ...

