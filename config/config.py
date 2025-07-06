import yaml
import random

class Config:
    def __init__(self, config_dict=None, default_config=None):
        # 使用默认配置初始化类对象
        if default_config:
            for key, value in default_config.items():
                if isinstance(value, dict):
                    value = Config(value)
                setattr(self, key, value)
        
        # 使用字典的内容覆盖初始化类对象
        if config_dict:
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    value = Config(value)
                setattr(self, key, value)

def load_default_config():
    # 定义 _C 的默认配置
    _C = {
        "BASE": [""],
        "DATA": {
            "BATCH_SIZE": 32,
            "DATA_PATH": "",
            "DATASET": "livec",
            "PATCH_NUM": 25,
            "IMG_SIZE": 224,
            "CROP_SIZE": (224, 224),
            "ZIP_MODE": False,
            "CACHE_MODE": "part",
            "PIN_MEMORY": True,
            "NUM_WORKERS": 4
        },
        "SET": {
            "COUNT": 1162,
            "TRAIN_INDEX": None,
            "TEST_INDEX": None
        },
        "MODEL": {
            "TYPE": "swin",
            "NAME": "swin_tiny_patch4_window7_224",
            "PRETRAINED": "",
            "RESUME": "",
            "NUM_CLASSES": 1,
            "DROP_RATE": 0.0,
            "DROP_PATH_RATE": 0.1,
            "VIT": {
                "PATCH_SIZE": 16,
                "EMBED_DIM": 384,
                "DEPTH": 12,
                "NUM_HEADS": 6,
                "MLP_RATIO": 4,
                "QKV_BIAS": True,
                "PRETRAINED": True,
                "PRETRAINED_MODEL_PATH": "",
                "CROSS_VALID": False,
                "CROSS_MODEL_PATH": ""
            },
            "HOR": {
                "PRETRAINED": True,
                "PRETRAINED_MODEL_PATH": ""
            },
            "COC": {
                "PRETRAINED": True,
                "PRETRAINED_MODEL_PATH": "",
                "CROSS_VALID": False,
                "CROSS_MODEL_PATH": ""
            }
        },
        "TRAIN": {
            "START_EPOCH": 0,
            "EPOCHS": 300,
            "WARMUP_EPOCHS": 20,
            "WEIGHT_DECAY": 0.05,
            "BASE_LR": 5e-4,
            "WARMUP_LR": 5e-7,
            "MIN_LR": 5e-6,
            "CLIP_GRAD": 5.0,
            "AUTO_RESUME": True,
            "ACCUMULATION_STEPS": 1,
            "USE_CHECKPOINT": False,
            "LR_SCHEDULER": {
                "NAME": "cosine",
                "DECAY_EPOCHS": 30,
                "DECAY_RATE": 0.1
            },
            "OPTIMIZER": {
                "NAME": "adamw",
                "EPS": 1e-8,
                "BETAS": (0.9, 0.999),
                "MOMENTUM": 0.9
            }
        },
        "TEST": {
            "SEQUENTIAL": False
        },
        "AMP_ENABLE": True,
        "AMP_OPT_LEVEL": "",
        "OUTPUT": "",
        "TAG": "default",
        "SAVE_FREQ": 1,
        "DISABLE_SAVE": False,
        "PRINT_FREQ": 10,
        "SEED": 0,
        "EVAL_MODE": False,
        "THROUGHPUT_MODE": False,
        "DEBUG_MODE": False,
        "EXP_INDEX": 0,
        "LOCAL_RANK": 0,
        "FUSED_WINDOW_PROCESS": True
    }
    return _C

def load_config(yaml_path):
    # 加载默认配置
    default_config = load_default_config()
    
    # 打开并加载 YAML 配置文件
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # 使用默认配置初始化，并覆盖更新
    config = Config(config_dict=config_dict, default_config=default_config)
    
    # 进一步对配置进行处理
    sel_num = list(range(0, config.SET.COUNT))
    random.shuffle(sel_num)
    config.SET.TRAIN_INDEX = sel_num[0: int(round(0.8 * len(sel_num)))]
    config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
    config.TRAIN.OPTIMIZER.EPS = float(config.TRAIN.OPTIMIZER.EPS)
    
    return config
