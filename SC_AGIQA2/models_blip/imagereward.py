# pip install image-reward
import ImageReward as RM
import clip
# import torch
# device = torch.device("cuda:3")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只让代码看到 cuda:3
model = RM.load("ImageReward-v1.0")
rewards = model.score("a cat", ["/home/Newdisk/lq/My-IQA/models_blip/cat.jpg", "/home/Newdisk/lq/My-IQA/models_blip/dog.jpg"])
print(rewards)