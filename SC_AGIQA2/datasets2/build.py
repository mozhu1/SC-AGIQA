import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms

from .iqa_dataset import *
from .samplers import IQAPatchDistributedSampler, SubsetRandomSampler


def build_transform(is_train, config):
    if config.DATA.DATASET == "koniq":
        if is_train:
            transform = transforms.Compose(
                [   #垂直翻转，本来没有的
                    transforms.RandomVerticalFlip(p=0.5), 
                    transforms.RandomHorizontalFlip(),
                    ###正常没有这一句
                    #transforms.RandomCrop(size=(224,224)),
                    ###正常下面这一句不注释
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    ###正常没有这一句
                    ###正常下面这一句不注释
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    #transforms.Resize((384, 384)),
                    transforms.RandomHorizontalFlip(),
                    #transforms.Lambda(lambda img: transforms.Resize((round(img.height // 1.2), round(img.width // 1.2)))(img)),
                    transforms.RandomCrop(size=(config.DATA.CROP_SIZE, config.DATA.CROP_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    #transforms.Resize((224, 224)),
                    transforms.RandomCrop(size=(config.DATA.CROP_SIZE, config.DATA.CROP_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((224, 224)),
                    #transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    #transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "agiqa3k":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    #transforms.Resize((384, 384)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    #transforms.Resize((384, 384)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
    if config.DATA.DATASET == "aigciqa2023":
        if is_train:
            transform = transforms.Compose(
                [
                    #transforms.Resize((512, 384)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    #transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
    if config.DATA.DATASET == "bid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "agiqa1k":
        if is_train:
            transform = transforms.Compose(
                [
                    #transforms.Resize((512, 384)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    #transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
    return transform


def build_IQA_dataset(config):
    print(config.DATA.DATASET)
    if config.DATA.DATASET == "koniq":
        train_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "uw":
        train_dataset = UWIQADATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = UWIQADATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livec":
        train_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "live":
        train_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),\
            is_train=False
        )
    elif config.DATA.DATASET == "tid2013":
        train_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "csiq":
        train_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "kadid":
        train_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "spaq":
        train_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "livefb":
        train_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "agiqa3k":
        train_dataset = AGIQA3K(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
            is_train=True
        )
        test_dataset = AGIQA3K(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
            is_train=False
        )
    elif config.DATA.DATASET == "bid":
        train_dataset = BIDDataset(
        config.DATA.DATA_PATH,
        config.SET.TRAIN_INDEX,
        config.DATA.PATCH_NUM,
        transform=build_transform(is_train=True, config=config),
    )
        test_dataset = BIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "aigciqa2023":
        train_dataset = AIGCIQA2023KDataset(
        config.DATA.DATA_PATH,
        config.SET.TRAIN_INDEX,
        config.DATA.PATCH_NUM,
        transform=build_transform(is_train=True, config=config),
    )
        test_dataset = AIGCIQA2023KDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "agiqa1k":
        train_dataset = AGIQA1K(
        config.DATA.DATA_PATH,
        config.SET.TRAIN_INDEX,
        config.DATA.PATCH_NUM,
        transform=build_transform(is_train=True, config=config),
    )
        test_dataset = AGIQA1K(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset

from torch.utils.data import DataLoader, Dataset, Sampler

# class GroupedSampler(Sampler):
#     def __init__(self, data_source, group_size):
#         """
#         data_source: 数据集
#         group_size: 每组的大小（例如 10）
#         """
#         self.data_source = data_source
#         self.group_size = group_size
#         self.num_groups = len(data_source) // group_size

#     def __iter__(self):
#         # 打乱组的顺序
#         group_indices = torch.randperm(self.num_groups)
#         for group_idx in group_indices:
#             # 获取当前组的索引
#             start = group_idx * self.group_size
#             end = start + self.group_size
#             # 返回当前组的所有索引
#             yield from range(start, end)

#     def __len__(self):
#         return len(self.data_source)

# def IQA_build_loader(config):
#     # 构建训练和验证数据集
#     dataset_train, dataset_val = build_IQA_dataset(config=config)

#     group_size = 2  # 每组 2 个 crop
#     sampler = GroupedSampler(dataset_train, group_size)

#     # 创建数据加载器
#     data_loader_train = DataLoader(
#         dataset_train,
#         batch_size=config.DATA.BATCH_SIZE,  # 批处理大小
#         sampler=sampler,  # 使用自定义采样器
#         num_workers=config.DATA.NUM_WORKERS,  # 工作线程数量
#         pin_memory=config.DATA.PIN_MEMORY,  # 锁页内存
#         drop_last=False,  # 丢弃最后一个不完整的批次
#     )

#     # 创建验证数据加载器
#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val,
#         batch_size=config.DATA.BATCH_SIZE,  # 批处理大小
#         shuffle=False,  # 验证时不打乱数据
#         num_workers=config.DATA.NUM_WORKERS,  # 工作线程数量
#         pin_memory=config.DATA.PIN_MEMORY,  # 锁页内存
#         drop_last=False,  # 保留最后一个不完整的批次
#     )

#     # 返回训练集、验证集和对应的数据加载器
#     return dataset_train, dataset_val, data_loader_train, data_loader_val
import random
import os
import torch
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def seed_worker(worker_id):
    setup_seed(20)
    worker_seed = torch.initial_seed() % 2**32  # 获取当前线程的随机种子
    #print(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def IQA_build_loader(config):
    setup_seed(20)
    # 构建训练和验证数据集
    dataset_train, dataset_val = build_IQA_dataset(config=config)

    # 创建训练数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,  # 批处理大小
        shuffle=True,  # 训练时打乱数据
        num_workers=config.DATA.NUM_WORKERS,  # 工作线程数量
        pin_memory=config.DATA.PIN_MEMORY,  # 锁页内存
        drop_last=True,  # 丢弃最后一个不完整的批次
        worker_init_fn=seed_worker,
    )

    # 创建验证数据加载器
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,  # 批处理大小
        shuffle=False,  # 验证时不打乱数据
        num_workers=config.DATA.NUM_WORKERS,  # 工作线程数量
        pin_memory=config.DATA.PIN_MEMORY,  # 锁页内存
        drop_last=False,  # 保留最后一个不完整的批次
        worker_init_fn=seed_worker,
    )

    # 返回训练集、验证集和对应的数据加载器
    return dataset_train, dataset_val, data_loader_train, data_loader_val