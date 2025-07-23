import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets2 import IQA_build_loader
from config.config import load_config
from models_blip.blip import BLIPRegressionModel,build_agiqa_model
from optimizer import build_optimizer
from scipy.stats import spearmanr, pearsonr
from utils.utils import (
    NativeScalerWithGradNormCount,
)
import random
import matplotlib.pyplot as plt
from lr_scheduler import build_scheduler
import torchmetrics
from thop import profile
from tqdm import tqdm  
import numpy as np
def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
def count_forward_flops(model, input_shape=(1, 3, 224, 224)):
    try:
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_shape).to(device)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        print(f"模型前向推理一次的 FLOPs：{flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        print("计算 FLOPs 时出现错误:", e)
def load_training_config(config_path):
    config = load_config(config_path)
    return config
def initialize_model_and_optimizer(config):
    device = torch.device("cuda")
    print(f"device:{device}")
    model=build_agiqa_model(config,device)
    model.blip_encoder.visual_encoder.eval()
    model.blip_encoder.reward.train()
    for name, parms in model.blip_encoder.reward.mlp.named_parameters():
        parms.requires_grad_(False)
    for name, parms in model.blip_encoder.reward.blip.named_parameters():
        if '_proj' in name:
            parms.requires_grad_(False)
    if model.blip_encoder.fix_rate > 0:
        text_fix_num = "layer.{}".format(int(12 * model.blip_encoder.fix_rate))
        image_fix_num = "blocks.{}".format(int(24 * model.blip_encoder.fix_rate))
        for name, parms in model.blip_encoder.reward.blip.text_encoder.named_parameters():
            parms.requires_grad_(False)
            if text_fix_num in name:
                break
        for name, parms in model.blip_encoder.reward.blip.visual_encoder.named_parameters():
            parms.requires_grad_(False)
            if image_fix_num in name:
                break
    count_model_params(model)
    print("模型初始化完成，设备为:", device)
    optimizer = build_optimizer(config, model)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, gamma=0.1)
    return model, optimizer, scheduler, device
def train_epoch(epoch, model, data_loader, optimizer, config, device, lr_scheduler):
    model.train()   
    model.blip_encoder.visual_encoder.eval()
    model.blip_encoder.reward.train()
    for name, parms in model.blip_encoder.reward.mlp.named_parameters():
        parms.requires_grad_(False)
    for name, parms in model.blip_encoder.reward.blip.named_parameters():
        if '_proj' in name:
            parms.requires_grad_(False)
    if model.blip_encoder.fix_rate > 0:
        text_fix_num = "layer.{}".format(int(12 * model.blip_encoder.fix_rate))
        image_fix_num = "blocks.{}".format(int(24 * model.blip_encoder.fix_rate))
        for name, parms in model.blip_encoder.reward.blip.text_encoder.named_parameters():
            parms.requires_grad_(False)
            if text_fix_num in name:
                break
        for name, parms in model.blip_encoder.reward.blip.visual_encoder.named_parameters():
            parms.requires_grad_(False)
            if image_fix_num in name:
                break
    running_loss = 0.0
    total_samples = 0
    num_steps = len(data_loader)
    with tqdm(data_loader, desc=f"Epoch {epoch}/{config.TRAIN.EPOCHS}", unit="batch") as pbar:
        for batch_idx, (images, labels,consis) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True).contiguous() , labels.to(device, non_blocking=True).contiguous()  
            with torch.autograd.detect_anomaly():
                outputs = model(images,consis)
                labels.unsqueeze_(dim=-1)
                loss = compute_loss(outputs, labels) / config.TRAIN.ACCUMULATION_STEPS 
                loss = loss.contiguous()  
                loss.backward() 
            if (batch_idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()  
                optimizer.zero_grad() 
                lr_scheduler.step_update(
                    (epoch * num_steps + batch_idx) // config.TRAIN.ACCUMULATION_STEPS
                )
            running_loss += loss.item() * images.size(0) * config.TRAIN.ACCUMULATION_STEPS
            total_samples += images.size(0)
            pbar.set_postfix({"Loss": loss.item() * config.TRAIN.ACCUMULATION_STEPS})
    avg_loss = running_loss / total_samples
    print(f"训练 Epoch [{epoch}/{config.TRAIN.EPOCHS}] 完成, 平均 Loss: {avg_loss:.4f}")
    return avg_loss
def compute_loss(preds, labels):
    criterion = nn.SmoothL1Loss()
    loss_smooth_l1 = criterion(preds,labels)
    return loss_smooth_l1
def setup_seed(args):
    valid_configs = {
        "agiqa1k.yaml": 1741929688,
        "aigciqa2023.yaml": 1742174487,
        "agiqa3k.yaml": 1742174487
    }
    config_name = args.config.split('/')[-1]
    if config_name not in valid_configs:
        raise ValueError(f"无效的配置文件: {config_name}，必须是 {list(valid_configs.keys())} 之一。")
    seed = valid_configs[config_name]
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(20)
    torch.cuda.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
def validate_epoch(model, data_loader, config, device,epoch):
    model.eval() 
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    best_pearson_avg=0
    with torch.no_grad(): 
        for images, labels,consis in data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)#,consis.to(device, non_blocking=True),consis
            outputs= model(images,consis)
            labels.unsqueeze_(dim=-1)
            loss = compute_loss(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.append(outputs)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0).cpu()  
    all_labels = torch.cat(all_labels, dim=0).cpu().unsqueeze(1)
    if config.DATA.PATCH_NUM > 1:
        ###截断最后一组patch_num防止报错
        valid_len = (all_preds.shape[0] // config.DATA.PATCH_NUM) * config.DATA.PATCH_NUM
        all_preds = all_preds[:valid_len]
        all_labels = all_labels[:valid_len]
        batch_size = valid_len // config.DATA.PATCH_NUM
        all_preds = all_preds.view(batch_size, config.DATA.PATCH_NUM, -1)
        all_labels = all_labels.view(batch_size, config.DATA.PATCH_NUM, -1)  
        avg_preds = all_preds.mean(dim=1)  
        avg_labels = all_labels.mean(dim=1)  
        avg_preds = avg_preds.squeeze()  
        avg_labels = avg_labels.squeeze()  
        spearman_avg = torchmetrics.functional.spearman_corrcoef(
            avg_preds.float().detach(), 
            avg_labels.float().detach()  
        ).item()
        pearson_avg = torchmetrics.functional.pearson_corrcoef(
            avg_preds.float().detach(), 
            avg_labels.float().detach() 
        ).item()
        if pearson_avg>best_pearson_avg:
            best_pearson_avg=pearson_avg
            #torch.save(model.state_dict(), f'./checkpoints_2023k/blip_model_weights{best_pearson_avg}.pth')

        print(f"Epoch:{epoch},平均分数的斯皮尔曼系数: {spearman_avg:.4f}, 皮尔逊相关系数: {pearson_avg:.4f}")
    spearman_corr = torchmetrics.functional.spearman_corrcoef(
        avg_preds.float().detach(), 
        avg_labels.float().detach()  
    ).item()
    pearson_corr = torchmetrics.functional.pearson_corrcoef(
        avg_preds.float().detach(), 
        avg_labels.float().detach() 
    ).item()
    avg_loss = running_loss / total_samples
    print(f"验证损失: {avg_loss:.4f}")
    return avg_loss, spearman_corr, pearson_corr

def plot_values(list1, list2):
    if not list1 or not list2:
        print("One of the input lists is empty.")
        return
    epochs = range(1, max(len(list1), len(list2)) + 1)
    plt.figure(figsize=(10, 5))  
    plt.plot(epochs[:len(list1)], list1, label='List 1', marker='o')  
    plt.plot(epochs[:len(list2)], list2, label='List 2', marker='x') 
    plt.title('Values vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig( './fig.pdf', format='pdf')
    plt.show()

def train_model(config):
    model, optimizer, scheduler, device = initialize_model_and_optimizer(config)
    train_dataset, test_dataset, data_loader_train, data_loader_val = IQA_build_loader(config)
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    spearman_corr_list = []
    pearson_corr_list = []
    train_loss_list = []  
    val_loss_list = []   
    best_pearson_epoch = 0
    best_spearman_epoch = 0
    best_spearman_corr = float('-inf') 
    best_pearson_corr = float('-inf')

    dataset_name = config.DATA.DATASET  
    log_filename = f"training_log_{dataset_name}.txt"
    with open(log_filename, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Batch Size: {config.DATA.BATCH_SIZE}\n")
        f.write(f"Accumulation Steps: {config.TRAIN.ACCUMULATION_STEPS}\n")  # 保存 ACCUMULATION_STEPS
        f.write("=" * 50 + "\n")
        for epoch in range(1, config.TRAIN.EPOCHS + 1):
            train_loss = train_epoch(epoch, model, data_loader_train, optimizer, config, device, lr_scheduler)
            train_loss_list.append(train_loss)  
            val_loss, spearman_corr, pearson_corr = validate_epoch(model, data_loader_val, config, device, epoch)
            val_loss_list.append(val_loss) 
            spearman_corr_list.append(spearman_corr)
            pearson_corr_list.append(pearson_corr)
            if pearson_corr > best_pearson_corr:
                best_pearson_epoch = epoch
                best_pearson_corr = pearson_corr
            if spearman_corr > best_spearman_corr:
                best_spearman_epoch = epoch
                best_spearman_corr = spearman_corr
            scheduler.step()
            epoch_info = (
                f"epoch: {epoch}, "
                f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}, "
                f"训练损失: {train_loss:.4f}, "
                f"验证损失: {val_loss:.4f}, "
                f"spearman: {best_spearman_corr:.4f}, "
                f"pearson: {best_pearson_corr:.4f}\n"
            )
            print(epoch_info)
            f.write(epoch_info)
        final_info = (
            f"最佳Spearman相关系数: {best_spearman_corr:.4f}, "
            f"最佳Pearson相关系数: {best_pearson_corr:.4f}, "
            f"最佳Spearman epoch: {best_spearman_epoch}, "
            f"最佳Pearson epoch: {best_pearson_epoch}\n"
        )
        print(final_info)
        f.write(final_info)
    plot_values(spearman_corr_list, pearson_corr_list)
def parse_args():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--config', type=str, default=f'/public/home/mozhu/SC_AGIQA/config/aigciqa2023.yaml', help="配置文件路径")
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    setup_seed(args)
    config = load_training_config(args.config)
    train_model(config)
if __name__ == "__main__":
    main()
